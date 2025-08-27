# convert_to_parquet.py
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import re
import os
import math

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
    """Convert transactional text to long-format Parquet (item, prob, txn_id) using cuDF.

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
        header=None,
        names=["items", "values"],
        dtype=["str", "str"],
    )
    
    # Clean trailing whitespace
    sep_esc = re.escape(sep)
    pattern = rf"[{sep_esc}\r\n ]+$"
    df["items"] = df["items"].str.replace(pattern, "", regex=True)
    df["values"] = df["values"].str.replace(pattern, "", regex=True)
    
    df['txn_id'] = (cudf.core.column.arange(0, len(df)) + 1).astype("uint32")

    df_long = cudf.DataFrame({
        'item': df['items'].str.split(sep).explode(),
        'prob': df['values'].str.split(sep).explode(),
        'txn_id': df['txn_id'].repeat(df['items'].str.count(sep) + 1),
    })
    
    df_long = df_long[df_long['item'] != '']
    df_long['prob'] = df_long['prob'].astype('uint32')
    
    df_long.to_parquet(output_path, compression=compression)
    
    variant = "fixed" if re.search(r"_fixed(\.|$)", input_path.name) else ("floating" if re.search(r"_floating(\.|$)", input_path.name) else "unknown")
    print(f"[convert] done method=cudf rows={len(df_long)} variant={variant} file={output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a transactional text file to a long-format Parquet file for cuFFIMiner."
    )
    parser.add_argument("inputFile", help="Path to the input text file (e.g., .csv, .txt)")
    parser.add_argument("outputFile", nargs='?', default=None, help="Optional. Path to the output Parquet file.")
    parser.add_argument("--sep", default="\t", help="Item/value separator inside a line (default: TAB)")
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec (zstd,snappy,gzip,none).")
    # Retaining old args for compatibility, but they are not used.
    parser.add_argument("--fast", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=["cudf"], default="cudf", help=argparse.SUPPRESS)
    parser.add_argument("--chunk-size", type=int, default=200000, help=argparse.SUPPRESS)
    
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