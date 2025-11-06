# convert_to_parquet.py
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import re

try:
    import cudf
    _CUDF_AVAILABLE = True
except ImportError:
    _CUDF_AVAILABLE = False
    raise SystemExit('cuDF not available; install RAPIDS to use this script.')

def convert_text_to_parquet(
    input_path_str: str,
    output_path_str: str = None,
    sep: str = "\t",
    compression: str = "zstd",
) -> str:
    """Convert transactional text to long-format Parquet (txn_id, item, prob) using cuDF.

    Returns: Parquet file path (str).
    """
    input_path = Path(input_path_str)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at '{input_path}'")

    if output_path_str:
        output_path = Path(output_path_str)
    else:
        output_path = input_path.with_suffix(".parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = cudf.read_csv(
        input_path,
        sep=":",
        names=["items", "values"],
        dtype=["str", "str"],
    )

    # Clean trailing whitespace
    sep_esc = re.escape(sep)
    pattern = rf"[{sep_esc}\r\n ]+$"
    df["items"] = df["items"].str.replace(pattern, "", regex=True)
    df["values"] = df["values"].str.replace(pattern, "", regex=True)

    # Add txn_id starting from 1 before exploding
    # Using index + 1 is efficient and correct here since we just read the file
    df['txn_id'] = df.index + 1
    df['txn_id'] = df['txn_id'].astype('uint32')


    # Explode items and values. 
    # We must explode both simultaneously to keep them aligned.
    # Since cuDF explode doesn't support multiple columns simultaneously in the same way pandas might,
    # we can use a common index to merge them back or explode them individually if they are guaranteed same length.
    # A safer way in cuDF for simultaneous explode is often to use `explode` on one and rely on index, but here's a robust approach:
    
    df_long = cudf.DataFrame({
        'txn_id': df['txn_id'].repeat(df['items'].str.count(sep) + 1), # repeat txn_id for each item
        'item': df['items'].str.split(sep).explode(),
        'prob': df['values'].str.split(sep).explode(),
    })
    # print(df_long)

    # Note: The above 'repeat' assumes 'items' and 'values' have same number of elements per row.
    # A simpler, more robust way often favored in cuDF if just exploding:
    
    # # Alternative robust explode:
    # # df_exploded = df.explode(['items', 'values']) # Only works if they are lists, currently they are strings to split
    # # So we split first:
    # df['items'] = df['items'].str.split(sep)
    # df['values'] = df['values'].str.split(sep)
    
    # Now explode both. cuDF's explode can take a list of columns to explode simultaneously
    # if they have matching list lengths in each row.
    # df_long = df.explode(['items', 'values']).rename(columns={'items': 'item', 'values': 'prob'})


    df_long = df_long[df_long['item'] != ''] # Remove empty items if any
    df_long['prob'] = df_long['prob'].astype('float64')
    
    # Reorder columns to preferred (txn_id, item, prob)
    df_long = df_long[['txn_id', 'item', 'prob']]

    print(f"[convert] method=cudf rows={len(df_long)} file={output_path}")
    
    df_long.to_parquet(output_path, compression=compression)
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a transactional text file to a long-format Parquet file for cuFFIMiner."
    )
    parser.add_argument("inputFile", help="Path to the input text file (e.g., .csv, .txt)")
    parser.add_argument("outputFile", nargs='?', default=None, help="Optional. Path to the output Parquet file.")
    parser.add_argument("--sep", default="\t", help="Item/value separator inside a line (default: TAB)")
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec (zstd,snappy,gzip,none).")
    
    args = parser.parse_args()
    try:
        convert_text_to_parquet(
            args.inputFile,
            args.outputFile,
            args.sep,
            compression=args.compression,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()