#!/usr/bin/env python3
"""
compare_polars_streamed_with_plots.py

This script conducts a performance comparison of Polars by generating three distinct
data shapes: 'square' (dense), 'triangle' (skewed), and 'empty' (a sparse L-shape).
It is explicitly optimized to use NULLs for missing values to ensure maximum
storage efficiency.

- 'csv' and 'parquet' (wide schema) explicitly manage NULLs/ragged rows.
- 'parquet-long' uses a sparse (row_index, value) schema, only storing non-nulls.

The `s` parameter controls the total number of cells (~s*s). The script writes
data in chunks to keep RAM usage low, times Polars read performance, and
generates plots for file sizes and read times.

This version has been optimized to use multiple threads for faster file generation.

Usage:
  python compare_polars_streamed_with_plots.py --sizes 64 128 --storage-limit-gib 8 --out out_all_shapes --clean --cores 4 --patterns square triangle --formats csv parquet --read-runs 5
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
from multiprocessing import Pool, Manager, Lock

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib

# ---------------- CONFIG ----------------
DEFAULT_SIZES = [64, 128, 160, 256, 320, 384, 512, 1024, 2048, 4096, 8192]
OUT_DIR = Path("out_all_shapes")
SUMMARY_NAME = "summary_streamed.csv"

# CHANGED: All available patterns and formats defined here for the CLI
ALL_PATTERNS = ["square", "triangle", "empty"]
ALL_FORMATS = ["csv", "parquet", "parquet-long"]

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
    Used for CSV and "wide" Parquet (parquet).
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


def long_generator(pattern: str, s: int):
    """
    Yields (row_index, value) tuples for all non-null entries.
    Used for "long" Parquet (parquet-long).
    """
    row_idx = 0
    for row_arr in row_generator(pattern, s):
        for val in row_arr:
            yield row_idx, val
        row_idx += 1


# ---------------- CSV WRITING (streamed) ----------------
def write_csv_stream(rows_iter, path: Path, compression: str | None):
    """
    Writes a CSV file by streaming rows.
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
    Writes a Parquet file (wide schema).
    """
    if schema_width == 0: return

    fields = [pa.field(f"c{j}", pa.uint32()) for j in range(schema_width)]
    schema = pa.schema(fields)

    with pq.ParquetWriter(str(path), schema=schema, compression=compression) as pqwriter:
        row_buffer = []
        for row in rows_iter:
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


def write_parquet_long_stream(long_rows_iter, path: Path, chunk_rows: int, compression: str | None):
    """
    Writes a Parquet file (long/sparse schema).
    """
    fields = [
        pa.field("row_index", pa.uint64()),
        pa.field("value", pa.uint32()),
    ]
    schema = pa.schema(fields)

    with pq.ParquetWriter(str(path), schema=schema, compression=compression) as pqwriter:
        row_index_buffer = []
        value_buffer = []
        for r_idx, val in long_rows_iter:
            row_index_buffer.append(r_idx)
            value_buffer.append(val)

            if len(row_index_buffer) >= chunk_rows:
                arrays = [
                    pa.array(row_index_buffer, type=pa.uint64()),
                    pa.array(value_buffer, type=pa.uint32()),
                ]
                pqwriter.write_table(pa.Table.from_arrays(arrays, schema=schema))
                row_index_buffer.clear()
                value_buffer.clear()

        if row_index_buffer:
            arrays = [
                pa.array(row_index_buffer, type=pa.uint64()),
                pa.array(value_buffer, type=pa.uint32()),
            ]
            pqwriter.write_table(pa.Table.from_arrays(arrays, schema=schema))


# --------------- READ TIMING via POLARS ----------------
# CHANGED: Function now takes num_runs, handles multiple readings, and averages them.
def time_polars_read(path: Path, fmt: str, num_runs: int) -> float:
    """
    Measures the time it takes for Polars to read a given file, running multiple
    times and averaging the result after removing outliers.
    """
    timings = []
    for _ in range(num_runs):
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
        elif fmt.startswith("parquet"):
            _ = pl.read_parquet(str(path))
        else:
            raise ValueError(f"Unknown format: {fmt}")
        timings.append(now_s() - t0)

    if not timings:
        return 0.0

    if len(timings) >= 3:
        # Remove the fastest and slowest times as simple outlier removal
        timings.sort()
        return np.mean(timings[1:-1])
    else:
        # Not enough data to remove outliers, just average
        return np.mean(timings)

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

    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'x', '+']
    
    # CHANGED: Updated y-axis label for read times plot
    for metric, ylabel in [('mib', 'Size (MiB)'), ('polars_read_s', 'Average Time (seconds)')]:
        fig, axes = plt.subplots(len(patterns), 1, figsize=(14, 7 * len(patterns)), squeeze=False)

        for i, pat in enumerate(patterns):
            ax = axes[i, 0]
            sub_df = df_pd[df_pd['pattern'] == pat]

            combos = [(fmt, comp) for fmt in sorted(sub_df['format'].unique())
                                for comp in sorted(sub_df[sub_df['format'] == fmt]['compression'].unique())]

            cmap = matplotlib.colormaps['viridis'].resampled(len(combos))
            for j, (fmt, comp) in enumerate(combos):
                comp_df = sub_df[(sub_df['format'] == fmt) & (sub_df['compression'] == comp)].sort_values('s')

                color = cmap(j)
                linestyle = line_styles[j % len(line_styles)]
                marker = markers[j % len(markers)]

                ax.plot(
                    comp_df['s'],
                    comp_df[metric],
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    label=f'{fmt}-{comp}'
                )

            title_prefix = "File Sizes" if metric == 'mib' else "Polars Read Times"
            ax.set_title(f"{title_prefix} — Pattern: {pat}")
            ax.set_xlabel('Size Parameter (s)')
            ax.set_ylabel(ylabel)
            ax.set_xscale('log', base=2)
            ax.legend()
            ax.grid(True, which="both", ls="--")

        fig.tight_layout()
        filename = 'file_sizes.png' if metric == 'mib' else 'read_times.png'
        fig.savefig(out_dir / filename, dpi=500)
        plt.close(fig)

    print(f"Wrote plots to {out_dir}")

# ---------------- MAIN EXPERIMENT DRIVER ----------------
# CHANGED: Worker function now accepts num_read_runs
def process_task(args):
    """A worker function to process a single file generation task."""
    pattern, s, fmt, comp, out_dir, chunk_mem_bytes, storage_limit_bytes, cumulative_bytes, lock, num_read_runs = args


    print(f"  Processing: s={s}, pattern='{pattern}', format='{fmt}', compression='{comp or 'none'}'")


    comp_str = "none" if comp is None else comp
    if fmt == "csv":
        ext = ".csv" if comp is None else (".csv.gz" if comp == "gzip" else ".zip")
        path = out_dir / f"{pattern}_s{s}{ext}"
    else:
        path = out_dir / f"{pattern}_s{s}.{fmt}.{comp_str}.parquet"

    print(f"    Writing {path.name}...", end="", flush=True)

    # if file exists, skip
    if path.exists():
        actual_size = file_size(path)
        print(f" exists, size: {human_bytes(actual_size)}")
        return {"pattern": pattern, "s": s, "cells": None, "format": fmt,
                "compression": comp_str, "path": str(path), "bytes": actual_size,
                "polars_read_s": time_polars_read(path, fmt, num_read_runs)}


    # with lock:
    #     if cumulative_bytes.value >= storage_limit_bytes:
    #         return None

    num_rows, max_cols, total_cells = get_shape_dims(pattern, s)


    if max_cols == 0 and fmt != "parquet-long":
        if total_cells == 0:
            print(f"    [SKIP] Pattern '{pattern}' with s={s} has no cells.")
            return None

    is_long_parquet = fmt == "parquet-long"

    if is_long_parquet:
        chunk_rows = max(1, chunk_mem_bytes // (8 + 4))
    elif fmt == "parquet":
        chunk_rows = max(1, chunk_mem_bytes // (4 * max_cols)) if max_cols > 0 else sys.maxsize
    else:
        chunk_rows = sys.maxsize

    try:
        if fmt == "csv":
            write_csv_stream(row_generator(pattern, s), path, comp)
        elif fmt == "parquet":
            write_parquet_stream(row_generator(pattern, s), path, max_cols, chunk_rows, comp)
        elif fmt == "parquet-long":
            write_parquet_long_stream(long_generator(pattern, s), path, chunk_rows, comp)
        else:
            raise ValueError(f"Unknown format: {fmt}")

        actual_size = file_size(path)

        with lock:
            if cumulative_bytes.value + actual_size > storage_limit_bytes:
                path.unlink()
                print(" limit exceeded, skipping.")
                return None
            cumulative_bytes.value += actual_size

        print(f" done: {human_bytes(actual_size)}")
        
        # CHANGED: Pass num_read_runs to the timing function
        t_read = time_polars_read(path, fmt, num_read_runs)
        return {"pattern": pattern, "s": s, "cells": total_cells, "format": fmt,
                "compression": comp_str, "path": str(path), "bytes": actual_size,
                "polars_read_s": t_read}
    except Exception as e:
        print(f" FAILED: {e}")
        return None

# CHANGED: Function now accepts lists of patterns/formats and the number of read runs
def run_experiment(sizes: list[int],
                   patterns_to_run: list[str],
                   formats_to_run: list[str],
                   out_dir: Path,
                   storage_limit_bytes: int,
                   chunk_mem_bytes: int,
                   num_cores: int | None,
                   num_read_runs: int):
    """Drives the main experiment loop using a process pool."""
    ensure_dir(out_dir)
    tasks = []

    format_compressions_map = {
        "csv": CSV_COMPRESSIONS,
        "parquet": PARQUET_COMPRESSIONS,
        "parquet-long": PARQUET_COMPRESSIONS,
    }

    for s in sizes:
        for pattern in patterns_to_run:
            for fmt in formats_to_run:
                for comp in format_compressions_map.get(fmt, []):
                    tasks.append((pattern, s, fmt, comp))

    with Manager() as manager:
        cumulative_bytes = manager.Value('i', 0)
        lock = manager.Lock()

        # CHANGED: Pass num_read_runs into the arguments for each worker
        pool_args = [(p, s, f, c, out_dir, chunk_mem_bytes, storage_limit_bytes, cumulative_bytes, lock, num_read_runs)
                     for p, s, f, c in tasks]

        with Pool(processes=num_cores) as pool:
            results = pool.map(process_task, pool_args)

    results = [r for r in results if r is not None]

    return pl.DataFrame(results)

# ---------------- COMMAND-LINE INTERFACE ----------------
def main():
    parser = argparse.ArgumentParser(description="Compare Polars read performance on dense and skewed data.")
    parser.add_argument("--sizes", nargs="*", type=int, help="List of size parameters (s).")
    parser.add_argument("--out", default=str(OUT_DIR), help="Output directory.")
    parser.add_argument("--storage-limit-gib", type=float, default=64.0, help="Global storage limit in GiB.")
    parser.add_argument("--chunk-mem-mib", type=int, default=512, help="Per-chunk memory cap in MiB.")
    parser.add_argument("--cores", type=int, default=None, help="Number of CPU cores to use. Defaults to all available.")
    parser.add_argument("--clean", action="store_true", help="Remove output directory before running.")
    
    # CHANGED: Added new arguments for selective runs and timing control
    parser.add_argument("--patterns", nargs="*", choices=ALL_PATTERNS, help="List of patterns to run.")
    parser.add_argument("--formats", nargs="*", choices=ALL_FORMATS, help="List of formats to run.")
    parser.add_argument("--read-runs", type=int, default=3, help="Number of times to run the read test for averaging.")
    
    args = parser.parse_args()

    out_dir = Path(args.out)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    sizes = args.sizes or DEFAULT_SIZES
    storage_limit_bytes = int(args.storage_limit_gib * 1024**3)
    chunk_mem_bytes = int(args.chunk_mem_mib * 1024**2)

    # CHANGED: Use user-provided patterns/formats or default to all
    patterns = args.patterns or ALL_PATTERNS
    formats = args.formats or ALL_FORMATS
    
    summary_path = out_dir / SUMMARY_NAME
    if summary_path.exists():
        print(f"Found existing summary file {summary_path}. Plotting from it.")
        df = pl.read_csv(str(summary_path))
        plot_results(df, out_dir)
        sys.exit(0)

    print("Running experiment in parallel.")
    print(f"Sizes: {sizes}")
    print(f"Patterns: {patterns}")
    print(f"Formats: {formats}")
    print(f"Read runs per file: {args.read_runs}")
    print(f"Storage limit: {human_bytes(storage_limit_bytes)}")
    print(f"Using up to {args.cores or os.cpu_count()} cores.")

    # CHANGED: Pass the selected patterns, formats, and read_runs to the experiment driver
    df_results = run_experiment(sizes, patterns, formats, out_dir, storage_limit_bytes, chunk_mem_bytes, args.cores, args.read_runs)

    if df_results.height > 0:
        df_sorted = df_results.sort(["s", "pattern", "format", "compression"])
        df_sorted.write_csv(str(summary_path))
        print(f"\nWrote summary to: {summary_path}")
        plot_results(df_sorted, out_dir)
    else:
        print("No files were produced.")

if __name__ == "__main__":
    main()