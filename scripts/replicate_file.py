#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import sys

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

def replicate_file(input_path_str: str, times: int, output_path_str: str = None) -> str:
    """Concatenate the file with itself *times* producing a dataset-level scaling factor (SF).

    Terminology update:
        SF (scaling factor) == number of concatenations (replications) performed.
        Output naming (if not provided):
            <stem>_SF{times}_floating<suffix>

    Examples:
        input.csv  + times=5 -> input_SF5_floating.csv

    If *times* == 1 we still create the canonical *SF1* copy (unless it already exists)
    so downstream code can rely uniformly on the naming pattern without special cases.

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

    text = path.read_text(encoding="utf-8")

    # Ensure copies are cleanly separated even if the file lacks a trailing newline.
    needs_nl = (len(text) > 0 and not text.endswith("\n"))
    base = text + ("\n" if needs_nl else "")
    repeated = base * times
    # If we added a newline for separation, but original had none, and you want to preserve
    # the final “no newline at EOF” semantics, you could strip the trailing one:
    if needs_nl and repeated.endswith("\n"):
        repeated = repeated[:-1]

    if output_path_str:
        out_path = Path(output_path_str)
    else:
        # New canonical naming
        out_path = path.with_name(f"{path.stem}_SF{times}_floating{path.suffix or ''}")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(repeated, encoding="utf-8")
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
