# import pandas as pd
# import cupy as cp
# import cudf
# import os

# # def csv_to_parquet(csv_path, parquet_path, sep):
# #     file = []
# #     with open(csv_path, 'r') as f:
# #         file = f.readlines()
        
# #     file = [line.strip().split(sep) for line in file]
    
# #     max_len = max([len(line) for line in file])
    
# #     df = pd.DataFrame(file, columns=[f'col_{i}' for i in range(max_len)])
    
# #     df.to_parquet(parquet_path)
    
# #     cudf_df = cudf.read_parquet(parquet_path)
# #     print(cudf_df.head())


# def csv_to_parquet_sparse(csv_path, parquet_path, sep=",", dtype=None, usecols=None):
#     # dtype: dict mapping col -> dtype to avoid slow type inference (optional but recommended)
#     # usecols: list of columns to read (optional)
#     gdf = cudf.read_csv(csv_path, sep=sep, dtype=dtype, usecols=usecols, header=None)
#     print(gdf.head())
#     # write directly to parquet on GPU
#     gdf.to_parquet(parquet_path, compression=None)  # compression can be 'snappy', 'gzip', etc.
#     return gdf  # optional: return gdf to inspect

# def csv_to_parquet(csv_path, parquet_path, sep=",", dtype=None, usecols=None):
#     # dtype: dict mapping col -> dtype to avoid slow type inference (optional but recommended)
#     # usecols: list of columns to read (optional)
#     gdf = cudf.read_csv(csv_path, sep=sep, dtype=dtype, usecols=usecols, header=None)
#     print(gdf.head())
#     # write directly to parquet on GPU
#     gdf.to_parquet(parquet_path, compression=None)  # compression can be 'snappy', 'gzip', etc.
#     return gdf  # optional: return gdf to inspect
    
    
# folder = "/export/home1/ltarun/cuda_pami/data/synthetic/transactional"
# files = os.listdir(folder)
# for file in files:
#     if file.endswith(".csv"):
#         csv_path = os.path.join(folder, file)
#         parquet_path = os.path.join(folder, file.replace(".csv", "_spa.parquet"))
#         csv_to_parquet(csv_path, parquet_path, sep=",")



import os
import math

# try cudf for GPU acceleration, otherwise use pandas (CPU)
try:
    import cudf as dfmod
    from cudf import DataFrame as _DF
    USING_CUDF = True
except Exception:
    import pandas as dfmod
    from pandas import DataFrame as _DF
    USING_CUDF = False

def csv_to_dense_parquet(csv_path, parquet_path, sep=",", dtype=None, usecols=None, header=None):
    """
    Read CSV and write dense parquet (one file that mirrors the CSV).
    Returns the DataFrame (cudf.DataFrame or pandas.DataFrame).
    """
    print("Using cudf:" if USING_CUDF else "Using pandas:")
    print(USING_CUDF)
    gdf = dfmod.read_csv(csv_path, sep=sep, dtype=dtype, usecols=usecols, header=header)
    # write dense parquet
    # compression=None disables compression; change to 'snappy' if desired
    out = parquet_path
    gdf.to_parquet(out, compression=None)
    return gdf

def create_sparse_versions(csv_path, out_folder=None, sep=",", dtype=None, usecols=None, header=None):
    """
    Create:
      - dense parquet: original turned to parquet (filename: <base>.dense.parquet)
      - sparse 3-column parquet: line_id, col, item (reconstructable) (filename: <base>.sparse3.parquet)
      - sparse 2-column parquet: line_id, item (compact) (filename: <base>.sparse2.parquet)

    Returns tuple(paths): (dense_path, sparse3_path, sparse2_path)
    """
    if out_folder is None:
        out_folder = os.path.dirname(csv_path) or "."

    base = os.path.splitext(os.path.basename(csv_path))[0]
    dense_path  = os.path.join(out_folder, base + ".dense.parquet")
    sparse3_path = os.path.join(out_folder, base + ".sparse3.parquet")
    sparse2_path = os.path.join(out_folder, base + ".sparse2.parquet")

    # Read CSV (no header by default)
    df = dfmod.read_csv(csv_path, sep=sep, dtype=dtype, usecols=usecols, header=header)

    # Write dense parquet
    df.to_parquet(dense_path, compression=None)

    # Ensure columns are simple 0..n-1 if header=None
    # Create line_id (transaction id) as integer range
    nrows = len(df)
    if USING_CUDF:
        # cudf: create a new column 'line_id' with integer range
        df = df.reset_index(drop=True)
        df["line_id"] = dfmod.Series(range(nrows), dtype="int64")
        value_vars = [c for c in df.columns if c != "line_id"]
        # melt (unpivot)
        melted = df.melt(id_vars="line_id", value_vars=value_vars,
                         var_name="col", value_name="item")
        # drop nulls and empty strings
        # For cudf, empty strings are a string, not NaN, so we filter both
        melted = melted.dropna(subset=["item"])
        # If some items can be empty string, remove them
        # Guard against dtype issues: convert item to string then filter if necessary
        if melted["item"].dtype == "object" or str(melted["item"].dtype).startswith("str"):
            melted = melted[melted["item"] != ""]
    else:
        # pandas
        df = df.reset_index(drop=True)
        df["line_id"] = dfmod.Series(range(nrows), dtype="int64")
        value_vars = [c for c in df.columns if c != "line_id"]
        melted = df.melt(id_vars="line_id", value_vars=value_vars,
                         var_name="col", value_name="item")
        melted = melted.dropna(subset=["item"])
        melted = melted[melted["item"].astype(str) != ""]

    # Save sparse 3-col (line_id, col, item)
    # Ensure column order
    sparse3 = melted[["line_id", "col", "item"]]
    sparse3.to_parquet(sparse3_path, compression=None)

    # Save sparse 2-col (line_id, item) - more compact transactional format
    sparse2 = sparse3[["line_id", "item"]]
    sparse2.to_parquet(sparse2_path, compression=None)

    print(f"Written dense -> {dense_path}")
    print(f"Written sparse3 -> {sparse3_path} (line_id, col, item)")
    print(f"Written sparse2 -> {sparse2_path} (line_id, item)")

    return dense_path, sparse3_path, sparse2_path

def reconstruct_from_sparse3(sparse3_path, out_csv_path=None, sep=","):
    """
    Reconstruct the original (dense) CSV from sparse3 (line_id, col, item).
    This recovers original column positions because we keep 'col'.
    """
    sparse = dfmod.read_parquet(sparse3_path)
    # make sure types are good
    # group by line_id and pivot by col -> item
    if USING_CUDF:
        # cudf: pivot currently limited; we'll do a groupby and create a list-of-tuples then rebuild via pandas for reliability
        import pandas as _pd
        # transfer to pandas for robust pivot (only if memory permits)
        pdf = sparse.to_pandas()
        # pivot (col values might be strings if original headers were strings)
        pivoted = pdf.pivot(index="line_id", columns="col", values="item")
        pivoted = pivoted.sort_index(axis=1)  # sort columns by col identifier
        # reset index to get line order
        dense_pdf = pivoted.reset_index(drop=True)
        if out_csv_path:
            dense_pdf.to_csv(out_csv_path, sep=sep, header=False, index=False)
        return dense_pdf
    else:
        # pandas path
        pivoted = sparse.pivot(index="line_id", columns="col", values="item")
        pivoted = pivoted.sort_index(axis=1)
        dense = pivoted.reset_index(drop=True)
        if out_csv_path:
            dense.to_csv(out_csv_path, sep=sep, header=False, index=False)
        return dense

def reconstruct_from_sparse2(sparse2_path, out_csv_path=None, sep=",", max_columns=None):
    """
    Reconstruct a dense CSV from sparse2 (line_id, item).
    Because 'col' is missing, we will re-create rows by collecting items for each line_id
    and placing them into columns 0..k-1 in the order they appear (order is preserved).
    If you need original column positions, use sparse3 instead.

    max_columns: optionally cap the number of columns (None = use max found).
    Returns the reconstructed DataFrame (pandas or cudf depending on availability).
    """
    sparse = dfmod.read_parquet(sparse2_path)

    # Ensure ordering: if the sparse format retained original row order when melted, items will be in reading order.
    # Group and aggregate into lists
    if USING_CUDF:
        # Transfer to pandas for list->columns expansion (more convenient)
        pdf = sparse.to_pandas()
        grouped = pdf.groupby("line_id")["item"].apply(list)
        max_len = max(len(lst) for lst in grouped)
        if max_columns is not None:
            max_len = min(max_len, max_columns)
        # convert to dataframe columns
        dense_pdf = dfmod = __import__('pandas').DataFrame(grouped.tolist())
        if out_csv_path:
            dense_pdf.to_csv(out_csv_path, sep=sep, header=False, index=False)
        return dense_pdf
    else:
        grouped = sparse.groupby("line_id")["item"].apply(list)
        max_len = max(len(lst) for lst in grouped)
        if max_columns is not None:
            max_len = min(max_len, max_columns)
        dense = __import__('pandas').DataFrame(grouped.tolist())
        if out_csv_path:
            dense.to_csv(out_csv_path, sep=sep, header=False, index=False)
        return dense

# Example helper for processing a whole folder
def process_folder(folder, sep=",", dtype=None, usecols=None, header=None):
    files = os.listdir(folder)
    for file in files:
        if file.endswith(".csv"):
            csv_path = os.path.join(folder, file)
            base = os.path.splitext(file)[0]
            dense_out = os.path.join(folder, base + ".dense.parquet")
            sparse3_out = os.path.join(folder, base + ".sparse3.parquet")
            sparse2_out = os.path.join(folder, base + ".sparse2.parquet")
            create_sparse_versions(csv_path, out_folder=folder, sep=sep, dtype=dtype, usecols=usecols, header=header)

# --- If run as script, example usage ---
if __name__ == "__main__":
        
    folder = "/export/home1/ltarun/cuda_pami/data/synthetic/transactional"
    files = os.listdir(folder)
    for file in files:
        if file.endswith(".csv"):
            csv_path = os.path.join(folder, file)
            parquet_path = os.path.join(folder, file.replace(".csv", ".parquet"))
            # csv_to_parquet(csv_path, parquet_path, sep=",")
            create_sparse_versions(csv_path, out_folder=folder, sep=",", header=None)
            # csv_to_dense_parquet(csv_path, parquet_path, sep=",", header=None)


