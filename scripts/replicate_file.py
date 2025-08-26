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

def main():
    ap = argparse.ArgumentParser(
        description="Write a new file containing N total copies of the original content; name as *_rep{N}.ext"
    )
    ap.add_argument("file", help="Path to the text file")
    ap.add_argument("times", help="Total copies, e.g., '2x' or '3' → rep2 / rep3")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    n = parse_times(args.times)
    text = path.read_text(encoding="utf-8")

    # Ensure copies are cleanly separated even if the file lacks a trailing newline.
    needs_nl = (len(text) > 0 and not text.endswith("\n"))
    base = text + ("\n" if needs_nl else "")
    repeated = base * n
    # If we added a newline for separation, but original had none, and you want to preserve
    # the final “no newline at EOF” semantics, you could strip the trailing one:
    if needs_nl and repeated.endswith("\n"):
        repeated = repeated[:-1]

    out_path = path.with_name(f"{path.stem}_rep{n}{path.suffix or ''}")
    out_path.write_text(repeated, encoding="utf-8")

    print(f"Wrote: {out_path} (contains {n}× the original content)")

if __name__ == "__main__":
    main()
