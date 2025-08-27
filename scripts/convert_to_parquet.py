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
import re
import os

# Optional high-performance dependencies
try:
    import polars as pl  # type: ignore
    _POLARS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _POLARS_AVAILABLE = False

import pandas as pd
import math
from typing import Optional
try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    _ARROW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ARROW_AVAILABLE = False


def convert_text_to_parquet(
    input_path_str: str,
    output_path_str: str = None,
    sep: str = "\t",
    fast: bool = False,
    method: Optional[str] = None,
    chunk_size: int = 200_000,
    compression: str = "zstd",
) -> str:
    """Convert transactional text to long-format Parquet (item, prob, txn_id).

    Modes (priority order if explicitly chosen):
      method='polars'  -> Polars lazy/parallel pipeline (fast, low memory, requires polars)
      method='stream'  -> Pure Python streaming writer (low peak memory, slower than Polars but steady)
      method='pandas'  -> Legacy pandas explode (simple, highest memory)
      method=None      -> Choose 'polars' if fast=True or file large & polars available else 'pandas'

    Args:
        input_path_str: Source text path.
        output_path_str: Destination parquet path (default: same stem .parquet).
        sep: Intra-line item/value separator (default TAB).
        fast: Backward-compat flag: if True and method not set, prefer polars.
        method: Explicit method selection ('polars'|'stream'|'pandas'|None).
        chunk_size: Transactions per batch for streaming mode.
        compression: Parquet compression codec (zstd|snappy|gzip|none...)
    Returns:
        Parquet file path (str).
    """
    input_path = Path(input_path_str)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at '{input_path}'")

    if output_path_str:
        output_path = Path(output_path_str)
    else:
        # Preserve *_SF{N}_fixed or *_SF{N}_floating portion and just change suffix
        output_path = input_path.with_suffix(".parquet")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal logging: final summary only (deferred)

    # Decide method
    if method is None:
        if fast and _POLARS_AVAILABLE:
            method = 'polars'
        else:
            # Auto-pick polars for large files if available
            if _POLARS_AVAILABLE and input_path.stat().st_size > 400 * 1024 * 1024:  # >400MB
                method = 'polars'
            else:
                method = 'pandas'
    method = method.lower()
    if method == 'polars' and not _POLARS_AVAILABLE:
        pass
        method = 'pandas'
    if method == 'stream' and not _ARROW_AVAILABLE:
        pass
        method = 'pandas'

    if method == 'polars':
        try:
            # Read colon separated two-column file lazily
            scan = pl.scan_csv(
                str(input_path),
                has_header=False,
                separator=":",
                new_columns=["items_str", "values_str"],
                try_parse_dates=False,
            )
            sep_esc = re.escape(sep)
            pattern = f"[{sep_esc}\r\n ]+$"
            # Generate a txn_id BEFORE explode so it replicates per transaction
            df_long = (
                scan
                .with_columns([
                    pl.col("items_str").str.replace(pattern, "").str.split(sep),
                    pl.col("values_str").str.replace(pattern, "").str.split(sep),
                    pl.arange(1, pl.count()+1).alias("txn_id")
                ])
                .select(["items_str", "values_str", "txn_id"])  # keep only needed
                .rename({"items_str": "items", "values_str": "values"})
                .explode(["items", "values"])  # long
                .filter(pl.col("items") != "")
                .rename({"items": "item", "values": "prob"})
                .with_columns([
                    pl.col("prob").cast(pl.UInt64),
                    pl.col("item").cast(pl.Utf8),
                    pl.col("txn_id").cast(pl.UInt32),
                ])
                .collect()
            )
            if df_long["prob"].max() <= 2**32 - 1:
                df_long = df_long.with_columns(pl.col("prob").cast(pl.UInt32))
            df_long.write_parquet(str(output_path), compression=compression, use_pyarrow=True)
            variant = "fixed" if re.search(r"_fixed(\.|$)", input_path.name) else ("floating" if re.search(r"_floating(\.|$)", input_path.name) else "unknown")
            print(f"[convert] done method=polars rows={df_long.height} variant={variant} file={output_path}")
            return str(output_path)
        except Exception as e:  # pragma: no cover
            # Silently fallback (verbosity minimized)
            method = 'pandas'

    if method == 'stream':
        # Memory-light streaming writer; build row groups in batches
    # Streaming start suppressed; only final summary will be printed
        writer = None
        total_rows = 0
        item_buf: list[str] = []
        prob_buf: list[int] = []
        txn_buf: list[int] = []
        flushes = 0
        with input_path.open('r') as f:
            for txn_id, line in enumerate(f, start=1):
                line = line.rstrip('\n\r')
                if not line:
                    continue
                try:
                    items_part, values_part = line.split(':', 1)
                except ValueError:
                    # Skip malformed
                    continue
                items = items_part.split(sep) if items_part else []
                values = values_part.split(sep) if values_part else []
                if len(items) != len(values):
                    continue
                for it, val in zip(items, values):
                    if not it:
                        continue
                    try:
                        ival = int(val)
                    except ValueError:
                        continue
                    item_buf.append(it)
                    prob_buf.append(ival)
                    txn_buf.append(txn_id)
                if len(txn_buf) >= chunk_size:
                    # Flush
                    table = pa.Table.from_arrays([
                        pa.array(item_buf, type=pa.string()),
                        pa.array(prob_buf, type=pa.uint32()),
                        pa.array(txn_buf, type=pa.uint32()),
                    ], names=["item", "prob", "txn_id"])
                    if writer is None:
                        writer = pq.ParquetWriter(str(output_path), table.schema, compression=compression)
                    writer.write_table(table)
                    total_rows += table.num_rows
                    flushes += 1
                    item_buf.clear(); prob_buf.clear(); txn_buf.clear()
            # Final flush
            if txn_buf:
                table = pa.Table.from_arrays([
                    pa.array(item_buf, type=pa.string()),
                    pa.array(prob_buf, type=pa.uint32()),
                    pa.array(txn_buf, type=pa.uint32()),
                ], names=["item", "prob", "txn_id"])
                if writer is None:
                    writer = pq.ParquetWriter(str(output_path), table.schema, compression=compression)
                writer.write_table(table)
                total_rows += table.num_rows
                flushes += 1
        if writer:
            writer.close()
    variant = "fixed" if re.search(r"_fixed(\.|$)", input_path.name) else ("floating" if re.search(r"_floating(\.|$)", input_path.name) else "unknown")
    print(f"[convert] done method=stream rows={total_rows} groups={flushes} variant={variant} file={output_path}")
    return str(output_path)

    # Legacy pandas path (method == 'pandas')
    try:
        df = pd.read_csv(
            input_path,
            sep=":",
            header=None,
            names=["items_str", "values_str"],
            dtype=str,
            engine="python",
        ).fillna("")
        sep_esc = re.escape(sep)
        pattern = rf"[{sep_esc}\r\n ]+$"
        df["items_str"] = df["items_str"].str.replace(pattern, "", regex=True)
        df["values_str"] = df["values_str"].str.replace(pattern, "", regex=True)
        df["items"] = df["items_str"].str.split(sep)
        df["values"] = df["values_str"].str.split(sep)
        df["txn_id"] = range(1, len(df) + 1)
        df_long = df.explode(["items", "values"])[["items", "values", "txn_id"]]
        df_long = df_long.rename(columns={"items": "item", "values": "prob"})
        df_long = df_long[df_long["item"] != ""]
        df_long["item"] = df_long["item"].astype("string")
        df_long["txn_id"] = df_long["txn_id"].astype("uint32")
        df_long["prob"] = pd.to_numeric(df_long["prob"]).astype("uint32")
    except ValueError as e:  # pragma: no cover
        raise ValueError(f"Could not convert a value to an integer. Check for non-numeric data. Details: {e}")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"An unexpected error occurred during conversion: {e}")

    # 7. Write to Parquet.
    variant = "fixed" if re.search(r"_fixed(\.|$)", input_path.name) else ("floating" if re.search(r"_floating(\.|$)", input_path.name) else "unknown")
    df_long.to_parquet(output_path, engine="pyarrow", index=False, compression=compression)
    print(f"[convert] done method=pandas rows={len(df_long)} variant={variant} file={output_path}")
    
    return str(output_path)

def main():
    """Main execution function for command-line use."""
    parser = argparse.ArgumentParser(
        description="Convert a transactional text file to a long-format Parquet file for cuFFIMiner."
    )
    parser.add_argument("inputFile", help="Path to the input text file (e.g., .csv, .txt)")
    parser.add_argument("outputFile", nargs='?', default=None, help="Optional. Path to the output Parquet file.")
    parser.add_argument("--sep", default="\t", help="Item/value separator inside a line (default: TAB)")
    parser.add_argument("--fast", action="store_true", help="(Deprecated) Shortcut to attempt Polars path; prefer --method polars.")
    parser.add_argument("--method", choices=["polars","pandas","stream"], default=None, help="Conversion method override.")
    parser.add_argument("--chunk-size", type=int, default=200000, help="Transactions per flush for stream method (default 200k).")
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec (zstd,snappy,gzip,none).")
    args = parser.parse_args()
    try:
        convert_text_to_parquet(
            args.inputFile,
            args.outputFile,
            args.sep,
            fast=args.fast,
            method=args.method,
            chunk_size=args.chunk_size,
            compression=args.compression,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()