#!/usr/bin/env python3
"""
Converts a transactional, pre-scaled integer dataset from a text file (CSV/TSV)
to the long-format Parquet file required by cuFFIMiner.

Input Text Format:
------------------
Each line represents a transaction. Items and their corresponding pre-scaled integer
values are separated by a colon ':'. Items/values themselves are separated by a
user-defined separator (e.g., a tab).

Example line (with tab separator):
itemA\titemB:100\t80

Output Parquet Format:
----------------------
A long-format table with three columns:
- item (string): The name of the item.
- prob (uint32): The pre-scaled integer probability/value.
- txn_id (uint32): The transaction identifier.

Example output for the line above:
| item  | prob | txn_id |
|-------|------|--------|
| itemA | 100  | 1      |
| itemB | 80   | 1      |

Dependencies:
-------------
This script requires pandas and pyarrow.
- pip install pandas pyarrow
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import re

def main():
    """Main execution function."""
    ap = argparse.ArgumentParser(
        description="Convert a transactional text file to a long-format Parquet file for cuFFIMiner."
    )
    ap.add_argument("inputFile", help="Path to the input text file (e.g., .csv, .txt)")
    ap.add_argument(
        "--sep",
        default="\t",
        help="The separator used between items/values within a transaction (default: tab)",
    )
    args = ap.parse_args()

    input_path = Path(args.inputFile)
    output_path = input_path.with_suffix(".parquet")
    print(f"Output will be written to '{output_path}'")

    if not input_path.exists():
        print(f"Error: Input file not found at '{input_path}'", file=sys.stderr)
        sys.exit(1)

    if output_path.suffix.lower() != ".parquet":
        print("Warning: Output file does not have a .parquet extension.", file=sys.stderr)

    print(f"Reading '{input_path}'...")

    try:
        # 1. Read the file as two columns separated by a colon.
        df = pd.read_csv(
            input_path,
            sep=":",
            header=None,
            names=["items_str", "values_str"],
            dtype=str,
            engine='python' # Slower but more robust for irregular separators
        )
        df.fillna("", inplace=True) # Handle empty lines gracefully

        # 2. Clean trailing separators and whitespace from both columns.
        sep_esc = re.escape(args.sep)
        pattern = rf"[{sep_esc}\r\n ]+$"
        df["items_str"] = df["items_str"].str.replace(pattern, "", regex=True)
        df["values_str"] = df["values_str"].str.replace(pattern, "", regex=True)

        # 3. Split the strings into lists of items and values.
        df["items"] = df["items_str"].str.split(args.sep)
        df["values"] = df["values_str"].str.split(args.sep)

        # 4. Add a 1-based transaction ID.
        df["txn_id"] = range(1, len(df) + 1)

        # 5. Explode the DataFrame to create the long format.
        # This works on pandas >= 1.3.0 and assumes lists in rows are of equal length.
        # It also handles rows where there is only one item (no separator).
        df_long = df.explode(["items", "values"])
        
        # Drop intermediate columns
        df_long = df_long[["items", "values", "txn_id"]]
        df_long = df_long.rename(columns={"items": "item", "values": "prob"})

        # Remove rows that might be empty after splitting
        df_long = df_long[df_long['item'] != '']

        print("Converting data types and validating...")
        
        # 6. Enforce the required data types for the Parquet file.
        df_long['item'] = df_long['item'].astype('string')
        df_long['txn_id'] = df_long['txn_id'].astype('uint32')
        # Convert 'prob' to a numeric type first, then to uint32.
        df_long['prob'] = pd.to_numeric(df_long['prob']).astype('uint32')

    except ValueError as e:
        print(f"Error: Could not convert a value to an integer. Check for non-numeric data in the values column.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    # 7. Write the final DataFrame to a Parquet file.
    print(f"Writing {len(df_long)} rows to '{output_path}'...")
    df_long.to_parquet(output_path, engine="pyarrow", index=False)

    print("Conversion successful.")


if __name__ == "__main__":
    main()