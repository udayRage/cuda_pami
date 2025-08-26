#!/usr/bin/env python3
import argparse
import re
from decimal import Decimal, getcontext, localcontext
from pathlib import Path

# Match numbers that contain a decimal point (".5", "0.75", "12.0", "123.456", etc.)
FLOAT_RE = re.compile(
    r"""
    (?P<num>
      (?<!\w)
      [+-]?           # optional sign
      (?:\d*\.\d+)    # must have a decimal point
      (?!\w)
    )
    """,
    re.VERBOSE,
)

def count_decimal_places(s: str) -> int:
    # Strip sign, split on '.', then strip trailing zeros
    s2 = s.lstrip("+-")
    _, frac = s2.split(".", 1)
    frac = frac.rstrip("0")
    return len(frac)

def main():
    ap = argparse.ArgumentParser(
        description="Scale all decimal numbers so they become integers; save as *_fixedpoint_p{N}.ext"
    )
    ap.add_argument("input_file", help="Path to the input text file")
    ap.add_argument("--dry-run", action="store_true", help="Show inferred precision and exit")
    args = ap.parse_args()

    in_path = Path(args.input_file)
    text = in_path.read_text(encoding="utf-8")

    floats = [m.group("num") for m in FLOAT_RE.finditer(text)]
    # If none found, we still produce p0 (scale=1) for consistency
    max_places = max((count_decimal_places(s) for s in floats), default=0)
    scale = 10 ** max_places

    print(f"Precision: p{max_places} (scale = {scale}) — {len(floats)} float(s) detected.")

    if args.dry_run:
        return

    getcontext().prec = 100
    with localcontext() as ctx:
        ctx.prec = 100
        scale_dec = Decimal(scale)

        def repl(m: re.Match) -> str:
            d = Decimal(m.group("num"))
            scaled = d * scale_dec
            # Convert to exact integer text
            return str(int(scaled.to_integral_value(rounding=ctx.rounding)))

        new_text = FLOAT_RE.sub(repl, text) if scale != 1 else text

    out_path = in_path.with_name(f"{in_path.stem}_fixedpoint_p{max_places}{in_path.suffix or ''}")
    out_path.write_text(new_text, encoding="utf-8")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
