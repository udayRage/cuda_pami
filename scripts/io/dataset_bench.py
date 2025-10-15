#!/usr/bin/env python3
"""
dataset_bench.py

A script to directly compare the performance of reading a standard CSV file
using Python's built-in library versus reading the same data in the optimized
Parquet format using the high-performance Polars library.

The script downloads a source CSV file from a URL, creates scaled versions of it,
and for each scaled CSV, it generates a corresponding Parquet file. It then
benchmarks the two distinct scenarios:
1. Standard Library reading the CSV.
2. Polars reading the Parquet file.

The results, including a plot named after the source file, are saved to the
output directory.

Usage:
  python dataset_bench.py \
    --url "https://raw.githubusercontent.com/pola-rs/polars-book/main/data/titanic.csv" \
    --scale-factors 1 2 4 8 16 32 \
    --out "benchmark_output" \
    --read-runs 5 \
    --clean
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import polars as pl
import requests

# ---------------- CONFIG ----------------
OUT_DIR = Path("benchmark_output")
SUMMARY_NAME = "summary_csv_vs_parquet.csv"

# ---------------- HELPERS ----------------
def now_s() -> float:
    """Returns the current time from a high-performance clock."""
    return time.perf_counter()

def human_bytes(n: int) -> str:
    """Converts a number of bytes into a human-readable string."""
    if n == 0: return "0 B"
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024: return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"

def ensure_dir(d: Path):
    """Ensures that a directory exists, creating it if necessary."""
    d.mkdir(parents=True, exist_ok=True)

def file_size(p: Path | str) -> int:
    """Returns the size of a file in bytes."""
    try:
        return os.path.getsize(str(p))
    except OSError:
        return 0

# --------------- READ TIMING FUNCTIONS ----------------
def _time_read_activity(path: Path, num_runs: int, read_func):
    """Generic timing harness for a read function."""
    timings = []
    for i in range(num_runs):
        print(f"      - Read run {i + 1}/{num_runs}...", end="\r")
        t0 = now_s()
        read_func(path)
        timings.append(now_s() - t0)
    print(" " * 30, end="\r")
    if not timings: return 0.0
    if len(timings) >= 3:
        timings.sort()
        return float(np.mean(timings[1:-1]))
    return float(np.mean(timings))

def time_polars_parquet_read(path: Path, num_runs: int) -> float:
    """Measures read time for a Parquet file using Polars."""
    return _time_read_activity(path, num_runs, pl.read_parquet)

def time_stdlib_csv_read(path: Path, num_runs: int) -> float:
    """Measures read time for a CSV file using Python's standard `csv` library."""
    def read(p: Path):
        if str(p).endswith(".zip"):
            with zipfile.ZipFile(p, 'r') as zf:
                name = zf.namelist()[0]
                with zf.open(name) as csvfile:
                    reader = csv.reader(io.TextIOWrapper(csvfile, 'utf-8'))
                    for _ in reader: pass
        else:
            with open(p, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for _ in reader: pass
    return _time_read_activity(path, num_runs, read)

# ---------------- PLOTTING ----------------
def plot_scaling_results(df: pl.DataFrame, out_dir: Path, source_filename: str):
    """Produces and saves plots for the benchmark comparison."""
    if df.height == 0:
        print("No scaling data to plot.")
        return

    df_pd = df.to_pandas()
    df_pd['csv_mib'] = df_pd['csv_bytes'] / (1024.0 * 1024.0)
    df_pd['parquet_mib'] = df_pd['parquet_bytes'] / (1024.0 * 1024.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # CHANGED: Added source_filename to the plot title
    fig.suptitle(f"Benchmark for: {source_filename}\nStandard Library (CSV) vs. Polars (Parquet)", fontsize=16)
    
    # --- Plot file sizes ---
    ax1.plot(df_pd['scale_factor'], df_pd['csv_mib'], marker='o', linestyle='-', label='CSV File Size')
    ax1.plot(df_pd['scale_factor'], df_pd['parquet_mib'], marker='s', linestyle='--', label='Parquet File Size')
    ax1.set_title('File Size Comparison (CSV vs. Parquet)')
    ax1.set_ylabel('Size (MiB)')
    ax1.grid(True, which="both", ls="--")
    ax1.legend()

    # --- Plot read times comparison ---
    ax2.plot(df_pd['scale_factor'], df_pd['stdlib_csv_read_s'], marker='^', linestyle=':', color='g', label='Standard Library (reading CSV)')
    ax2.plot(df_pd['scale_factor'], df_pd['polars_parquet_read_s'], marker='s', linestyle='--', color='r', label='Polars (reading Parquet)')
    ax2.set_title('Read Time Comparison')
    ax2.set_ylabel('Average Time (seconds)')
    ax2.set_xlabel('Scale Factor')
    ax2.grid(True, which="both", ls="--")
    ax2.legend()
    
    ax2.set_xscale('log', base=2)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xticks(df_pd['scale_factor'])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # CHANGED: Create a safe, dynamic filename for the plot image
    safe_filename = re.sub(r'[^\w\.\-]', '_', source_filename)
    plot_path = out_dir / f'benchmark_{safe_filename}.png'
    
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"\nWrote scaling plot to: {plot_path}")

# ---------------- MAIN BENCHMARK DRIVER ----------------
def benchmark_scaled_file(url: str, scale_factors: list[int], num_read_runs: int, out_dir: Path):
    """
    Downloads a CSV, creates scaled CSVs and corresponding Parquet files,
    and benchmarks their read times. Returns results and the source filename.
    """
    if not url.lower().endswith(('.csv', '.csv.zip', '.csv.gz')):
        print(f"Error: The provided URL must point to a CSV file. URL: {url}", file=sys.stderr)
        return None, None

    try:
        original_filename = url.split("/")[-1]
        original_path = out_dir / original_filename
        print(f"Downloading source CSV from {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(original_path, 'wb') as f: shutil.copyfileobj(r.raw, f)
        print("Download complete.")

        results = []
        for sf in scale_factors:
            print(f"\n--- Processing Scale Factor {sf}x ---")
            
            scaled_csv_path = out_dir / f"scaled_{sf}x_{original_filename}"
            print(f"  - Creating scaled CSV: {scaled_csv_path.name}")
            with open(scaled_csv_path, "wb") as scaled_f:
                with open(original_path, "rb") as original_f:
                    header = original_f.readline()
                    scaled_f.write(header)
                    shutil.copyfileobj(original_f, scaled_f)
                    for _ in range(sf - 1):
                        original_f.seek(0); original_f.readline()
                        shutil.copyfileobj(original_f, scaled_f)
            csv_size = file_size(scaled_csv_path)
            print(f"    - CSV size: {human_bytes(csv_size)}")

            parquet_path = scaled_csv_path.with_suffix(".parquet")
            print(f"  - Generating Parquet: {parquet_path.name}")
            pl.read_csv(scaled_csv_path).write_parquet(parquet_path)
            parquet_size = file_size(parquet_path)
            print(f"    - Parquet size: {human_bytes(parquet_size)}")

            print(f"  - Benchmarking stdlib `csv` ({num_read_runs} runs)...")
            t_stdlib = time_stdlib_csv_read(scaled_csv_path, num_read_runs)
            print(f"    - Stdlib `csv` average time: {t_stdlib:.4f}s")
            
            print(f"  - Benchmarking Polars Parquet ({num_read_runs} runs)...")
            t_polars = time_polars_parquet_read(parquet_path, num_read_runs)
            print(f"    - Polars Parquet average time: {t_polars:.4f}s")

            results.append({
                "scale_factor": sf, "csv_bytes": csv_size, "parquet_bytes": parquet_size,
                "stdlib_csv_read_s": t_stdlib, "polars_parquet_read_s": t_polars,
            })

    except requests.exceptions.RequestException as e:
        print(f"\nError: Failed to download file. {e}", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"\nError: An unexpected error occurred. {e}", file=sys.stderr)
        return None, None

    # CHANGED: Return the original filename along with the results
    return pl.DataFrame(results), original_filename

# ---------------- COMMAND-LINE INTERFACE ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Polars (reading Parquet) vs. stdlib (reading CSV) by scaling a dataset from a URL.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", type=str, required=True, help="URL of the source CSV dataset to benchmark.")
    parser.add_argument("--scale-factors", nargs="+", type=int, required=True, help="A list of integer scale factors (e.g., 1 2 4 8).")
    parser.add_argument("--out", default=str(OUT_DIR), help=f"Output directory for files and results (default: {OUT_DIR}).")
    parser.add_argument("--read-runs", type=int, default=3, help="Number of times to read each file for averaging (default: 3).")
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before running.")
    
    args = parser.parse_args()
    out_dir = Path(args.out)

    if args.read_runs < 1:
        print("Error: --read-runs must be at least 1.", file=sys.stderr)
        sys.exit(1)

    if args.clean and out_dir.exists():
        print(f"Removing existing output directory: {out_dir}")
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    # CHANGED: Unpack both the results and the source filename
    df_results, source_filename = benchmark_scaled_file(args.url, sorted(args.scale_factors), args.read_runs, out_dir)

    if df_results is not None and df_results.height > 0:
        summary_path = out_dir / f"summary_{source_filename}.csv"
        df_results.write_csv(str(summary_path))
        print(f"\n--- Benchmark Summary for {source_filename} ---")
        print(df_results)
        print(f"Results saved to: {summary_path}")
        
        # CHANGED: Pass the source filename to the plotting function
        plot_scaling_results(df_results, out_dir, source_filename)
    else:
        print("\nBenchmark did not produce any results.")
        sys.exit(1)

if __name__ == "__main__":
    main()