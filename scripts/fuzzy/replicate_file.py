#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import re
import sys
from typing import Optional

# Define a reasonable buffer size for reading/writing chunks
# 64KB is often a good balance for disk I/O, but you can adjust this.
BUFFER_SIZE = 64 * 1024 

def parse_times(xstr: str) -> int:
    """
    Accepts '2x', '3x', '5x' or plain integers like '2'.
    Returns total copies N (>=1).
    """
    m = re.fullmatch(r"\s*(\d+)\s*(x|X)?\s*", xstr)
    if not m:
        raise ValueError(f"Bad multiplier: {xstr!r} (try '2x' or '3')")
    n = int(m.group(1))
    if n < 1:
        raise ValueError("Multiplier must be >= 1")
    return n

def replicate_file(input_path_str: str, times: int, output_path_str: Optional[str] = None) -> str:
    """Concatenate the file with itself *times* producing a dataset-level scaling factor (SF) using chunked I/O for speed and low memory usage.

    Parameters
    ----------
    input_path_str : str
        Path to source text file.
    times : int
        Total number of concatenations (>=1).
    output_path_str : str, optional
        Explicit output path. If given, overrides automatic naming.

    Returns
    -------
    str
        Path to replicated (floating) file.
    """
    path = Path(input_path_str)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if times < 1:
        raise ValueError("Multiplier must be >= 1")

    if output_path_str:
        out_path = Path(output_path_str)
    else:
        # Canonical naming
        out_path = path.with_name(f"{path.stem}_SF{times}_floating{path.suffix or ''}")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # if output file exists, remove it
    if out_path.exists():
        return str(out_path)

    # --- Optimized Replication Logic using binary I/O ---
    
    # 1. Check for trailing newline only if SF > 1.
    needs_nl_separator = False
    if times > 1:
        try:
            # Check last byte. We open in binary ('rb') mode for this.
            with open(path, 'rb') as f:
                f.seek(-1, os.SEEK_END)
                last_byte = f.read(1)
                # 0x0A is the byte value for '\n' (LF)
                if last_byte != b'\n': 
                    needs_nl_separator = True
        except OSError:
             # Handle empty files or other OS read errors
             pass

    # 2. Replicate the file content using a binary buffer
    # We use binary mode ('wb') for faster I/O and direct byte copying.
    try:
        with open(out_path, 'wb') as outfile:
            for i in range(times):
                with open(path, 'rb') as infile:
                    while True:
                        # Read and write chunks
                        chunk = infile.read(BUFFER_SIZE)
                        if not chunk:
                            break
                        outfile.write(chunk)
                
                # Add newline separator if needed, but only if it's not the last copy.
                if needs_nl_separator and (i < times - 1):
                    outfile.write(b'\n')
    
    except Exception as e:
        # Clean up partial output file on error
        if out_path.exists():
            out_path.unlink()
        raise e

    print(f"[replicate] Wrote: {out_path} (SF={times})")
    return str(out_path)

def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Write a new file containing N total copies of the original content."
    )
    parser.add_argument("file", help="Path to the text file.")
    parser.add_argument("times", help="Total copies, e.g., '2x' or '3'.")
    parser.add_argument("output_file", nargs='?', default=None, help="Optional. Path to the output file.")
    args = parser.parse_args()

    try:
        n = parse_times(args.times)
        replicate_file(args.file, n, args.output_file)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()