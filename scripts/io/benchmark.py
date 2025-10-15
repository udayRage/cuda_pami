#!/usr/bin/env python3
"""
compare_polars_streamed_with_plots.py

This script conducts a performance comparison of Polars by generating three distinct
data shapes: 'square' (dense), 'triangle' (skewed), and 'empty' (a sparse L-shape).
It is explicitly optimized to use NULLs for missing values to ensure maximum
storage efficiency.

- For CSV, it generates ragged files where missing values are implicitly NULL.
- For Parquet, it pads rows with Python's `None`, which is written as a
  highly efficient NULL value.

The `s` parameter controls the total number of cells (~s*s). The script writes
data in chunks to keep RAM usage low, times Polars read performance, and
generates plots for file sizes and read times.

Usage:
  python compare_polars_streamed_with_plots.py --sizes 64 --storage-limit-gib 8 --out out_all_shapes --clean
"""
from __future__ import annotations

import argparse
import gzip
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------- CONFIG ----------------
DEFAULT_SIZES = [2, 4, 8, 16, 32, 64, 128, 256]
OUT_DIR = Path("out_all_shapes")
SUMMARY_NAME = "summary_streamed.csv"

CSV_COMPRESSIONS = [None, "gzip", "zip"]
PARQUET_COMPRESSIONS = [None, "snappy", "zstd", "lz4"]

# Memory for a single chunk (bytes) — controls how many rows per chunk:
DEFAULT_CHUNK_MEM_BYTES = 512 * 1024 * 1024  # 512 MiB

# Defaults for safety
DEFAULT_STORAGE_LIMIT_BYTES = 64 * 1024**3  # 64 GiB

# ---------------- HELPERS ----------------
def now_s() -> float:
    """Returns the current time from a high-performance clock."""
    return time.perf_counter()

def human_bytes(n: int) -> str:
    """Converts a number of bytes into a human-readable string."""
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PiB"

def ensure_dir(d: Path):
    """Ensures that a directory exists, creating it if necessary."""
    d.mkdir(parents=True, exist_ok=True)

def file_size(p: Path | str) -> int:
    """Returns the size of a file in bytes."""
    try:
        return os.path.getsize(str(p))
    except OSError:
        return 0

# ---------------- SHAPE LOGIC ----------------
def get_shape_dims(pattern: str, s: int) -> tuple[int, int, int]:
    """
    Calculates the dimensions for a given pattern and size `s`.

    Returns:
        A tuple of (number of rows, maximum number of columns, total cells).
    """
    target_cells = s * s
    if pattern == "square":
        return s, s, target_cells

    if pattern == "triangle":
        # Solve for k where k*(k+1)/2 is close to s*s
        if target_cells == 0: return 0, 0, 0
        k = int((-1 + math.sqrt(1 + 8 * target_cells)) / 2)
        total_cells = k * (k + 1) // 2
        return k, k, total_cells

    if pattern == "empty":
        # L-shape: one long horizontal bar and a tall single-element stem.
        if target_cells == 0: return 0, 0, 0
        h_bar_len = target_cells // 2
        v_stem_len = target_cells - h_bar_len
        num_rows = 1 + v_stem_len
        max_cols = h_bar_len
        return num_rows, max_cols, target_cells

    raise ValueError(f"Unknown pattern: {pattern}")


# ---------------- GENERATORS (row-streaming) ----------------
def row_generator(pattern: str, s: int):
    """
    Yields rows as numpy uint32 arrays. Row lengths may vary.
    """
    num_rows, max_cols, _ = get_shape_dims(pattern, s)
    val_counter = 0

    if pattern == "square":
        for _ in range(num_rows):
            yield np.arange(val_counter, val_counter + max_cols, dtype=np.uint32)
            val_counter += max_cols

    elif pattern == "triangle":
        for r in range(num_rows):
            row_len = max_cols - r
            yield np.arange(val_counter, val_counter + row_len, dtype=np.uint32)
            val_counter += row_len
            
    elif pattern == "empty":
        # First row is the long horizontal bar
        yield np.arange(val_counter, val_counter + max_cols, dtype=np.uint32)
        val_counter += max_cols
        # Subsequent rows are the single-element vertical stem
        for _ in range(num_rows - 1):
            yield np.array([val_counter], dtype=np.uint32)
            val_counter += 1
            
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


# ---------------- CSV WRITING (streamed) ----------------
def write_csv_stream(rows_iter, path: Path, compression: str | None):
    """
    Writes a CSV file by streaming rows. By writing fewer values for shorter
    rows (a "ragged" file), data readers correctly interpret the missing
    fields as NULL, which is the most storage-efficient method for CSV.
    """
    if compression == "zip":
        tmp_csv_path = path.with_suffix(".csv.tmp")
        with open(tmp_csv_path, "wb") as f:
            for arr in rows_iter:
                line = b",".join(map(lambda x: str(int(x)).encode(), arr)) + b"\n"
                f.write(line)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            arcname = tmp_csv_path.with_suffix("").name
            zf.write(tmp_csv_path, arcname=arcname)
        tmp_csv_path.unlink()
    else:
        opener = gzip.open if compression == "gzip" else open
        with opener(path, "wb") as f:
            for arr in rows_iter:
                line = b",".join(map(lambda x: str(int(x)).encode(), arr)) + b"\n"
                f.write(line)

# ---------------- PARQUET WRITING (streamed row-groups) ----------------
def write_parquet_stream(rows_iter, path: Path, schema_width: int, chunk_rows: int, compression: str | None):
    """
    Writes a Parquet file. It explicitly pads ragged rows with Python's `None`
    value, which pyarrow writes as a highly efficient NULL representation.
    """
    if schema_width == 0: return
    
    fields = [pa.field(f"c{j}", pa.uint32()) for j in range(schema_width)]
    schema = pa.schema(fields)

    with pq.ParquetWriter(str(path), schema=schema, compression=compression) as pqwriter:
        row_buffer = []
        for row in rows_iter:
            # Pad the shorter row with `None`, which becomes NULL in Parquet.
            padded_row = list(row) + [None] * (schema_width - len(row))
            row_buffer.append(padded_row)

            if len(row_buffer) >= chunk_rows:
                transposed_cols = list(zip(*row_buffer))
                arrays = [pa.array(col, type=pa.uint32()) for col in transposed_cols]
                pqwriter.write_table(pa.Table.from_arrays(arrays, schema=schema))
                row_buffer.clear()

        if row_buffer:
            transposed_cols = list(zip(*row_buffer))
            arrays = [pa.array(col, type=pa.uint32()) for col in transposed_cols]
            pqwriter.write_table(pa.Table.from_arrays(arrays, schema=schema))

# --------------- READ TIMING via POLARS ----------------
def time_polars_read(path: Path, fmt: str) -> float:
    """Measures the time it takes for Polars to read a given file."""
    t0 = now_s()
    if fmt == "csv":
        p_str = str(path)
        if p_str.endswith(".zip"):
            with zipfile.ZipFile(p_str, 'r') as zf, tempfile.TemporaryDirectory() as td:
                name = zf.namelist()[0]
                zf.extract(name, td)
                _ = pl.read_csv(str(Path(td) / name), has_header=False)
        else:
            _ = pl.read_csv(p_str, has_header=False)
    elif fmt == "parquet":
        _ = pl.read_parquet(str(path))
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return now_s() - t0

# ---------------- PLOTTING ----------------
def plot_results(df: pl.DataFrame, out_dir: Path):
    """Produces and saves plots for file sizes and read times."""
    ensure_dir(out_dir)
    if df.height == 0:
        print("No data to plot.")
        return

    df_pd = df.to_pandas()
    df_pd['mib'] = df_pd['bytes'] / (1024.0 * 1024.0)
    
    patterns = sorted(df_pd['pattern'].unique())

    for metric, ylabel in [('mib', 'Size (MiB)'), ('polars_read_s', 'Time (seconds)')]:
        fig, axes = plt.subplots(len(patterns), 1, figsize=(14, 7 * len(patterns)), squeeze=False)
        for i, pat in enumerate(patterns):
            ax = axes[i, 0]
            sub_df = df_pd[df_pd['pattern'] == pat]
            for fmt in sorted(sub_df['format'].unique()):
                fmt_df = sub_df[sub_df['format'] == fmt]
                for comp in sorted(fmt_df['compression'].unique()):
                    comp_df = fmt_df[fmt_df['compression'] == comp].sort_values('s')
                    ax.plot(comp_df['s'], comp_df[metric], marker='o', linestyle='-', label=f'{fmt}-{comp}')
            
            title_prefix = "File Sizes" if metric == 'mib' else "Polars Read Times"
            ax.set_title(f"{title_prefix} — Pattern: {pat}")
            ax.set_xlabel('Size Parameter (s)')
            ax.set_ylabel(ylabel)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.legend(); ax.grid(True, which="both", ls="--")
        
        fig.tight_layout()
        filename = 'file_sizes.png' if metric == 'mib' else 'read_times.png'
        fig.savefig(out_dir / filename, dpi=500)
        plt.close(fig)

    print(f"Wrote plots to {out_dir}")

# ---------------- MAIN EXPERIMENT DRIVER ----------------
def run_experiment(sizes, out_dir: Path, storage_limit_bytes: int, chunk_mem_bytes: int):
    """Drives the main experiment loop."""
    ensure_dir(out_dir)
    results = []
    cumulative_bytes = 0
    patterns_to_run = ["square", "triangle", "empty"]

    for s in sizes:
        if cumulative_bytes >= storage_limit_bytes: break
        print(f"\n=== s={s} ===")

        for pattern in patterns_to_run:
            if cumulative_bytes >= storage_limit_bytes: break
            
            num_rows, max_cols, total_cells = get_shape_dims(pattern, s)
            raw_bytes = total_cells * 4
            
            print(f"  Pattern '{pattern}': {num_rows} rows, {max_cols} max_cols, {total_cells:,} cells ({human_bytes(raw_bytes)})")

            if max_cols == 0:
                print("    [SKIP] Pattern has no columns.")
                continue

            chunk_rows = max(1, chunk_mem_bytes // (4 * max_cols)) if max_cols > 0 else sys.maxsize

            for fmt, compressions in [("csv", CSV_COMPRESSIONS), ("parquet", PARQUET_COMPRESSIONS)]:
                for comp in compressions:
                    if cumulative_bytes >= storage_limit_bytes: break
                    comp_str = "none" if comp is None else comp

                    if fmt == "csv":
                        ext = ".csv" if comp is None else (".csv.gz" if comp == "gzip" else ".zip")
                        path = out_dir / f"{pattern}_s{s}{ext}"
                    else:
                        path = out_dir / f"{pattern}_s{s}.{comp_str}.parquet"

                    print(f"    Writing {fmt.upper()} ({comp_str})...", end="", flush=True)
                    try:
                        if fmt == "csv":
                            write_csv_stream(row_generator(pattern, s), path, comp)
                        else:
                            write_parquet_stream(row_generator(pattern, s), path, max_cols, chunk_rows, comp)
                        
                        actual_size = file_size(path)
                        if cumulative_bytes + actual_size > storage_limit_bytes:
                            path.unlink(); print(" limit exceeded, skipping."); continue
                        
                        cumulative_bytes += actual_size
                        print(f" done: {human_bytes(actual_size)}")
                        
                        t_read = time_polars_read(path, fmt)
                        results.append({"pattern": pattern, "s": s, "cells": total_cells, "format": fmt, 
                                        "compression": comp_str, "path": str(path), "bytes": actual_size, 
                                        "polars_read_s": t_read})
                    except Exception as e:
                        print(f" FAILED: {e}")
                
    if cumulative_bytes >= storage_limit_bytes:
        print("\nReached global storage limit. Stopping.")
    return pl.DataFrame(results), cumulative_bytes

# ---------------- COMMAND-LINE INTERFACE ----------------
def main():
    parser = argparse.ArgumentParser(description="Compare Polars read performance on dense and skewed data.")
    parser.add_argument("--sizes", nargs="*", type=int, help="List of size parameters (s).")
    parser.add_argument("--out", default=str(OUT_DIR), help="Output directory.")
    parser.add_argument("--storage-limit-gib", type=float, default=64.0, help="Global storage limit in GiB.")
    parser.add_argument("--chunk-mem-mib", type=int, default=512, help="Per-chunk memory cap in MiB.")
    parser.add_argument("--clean", action="store_true", help="Remove output directory before running.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    sizes = args.sizes or DEFAULT_SIZES
    storage_limit_bytes = int(args.storage_limit_gib * 1024**3)
    chunk_mem_bytes = int(args.chunk_mem_mib * 1024**2)
    
    summary_path = out_dir / SUMMARY_NAME
    if summary_path.exists():
        print(f"Found existing summary file {summary_path}. Plotting from it.")
        df = pl.read_csv(str(summary_path))
        plot_results(df, out_dir)
        sys.exit(0)

    print("Running experiment.")
    print(f"Sizes: {sizes}")
    print(f"Storage limit: {human_bytes(storage_limit_bytes)}")
    
    df_results, used_bytes = run_experiment(sizes, out_dir, storage_limit_bytes, chunk_mem_bytes)

    if df_results.height > 0:
        df_sorted = df_results.sort(["s", "pattern", "format", "compression"])
        df_sorted.write_csv(str(summary_path))
        print(f"\nWrote summary to: {summary_path} (Total space used: {human_bytes(used_bytes)})")
        plot_results(df_sorted, out_dir)
    else:
        print("No files were produced.")

if __name__ == "__main__":
    main()
