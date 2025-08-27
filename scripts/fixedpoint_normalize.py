#!/usr/bin/env python3
import argparse
import re
import shutil
from decimal import Decimal, getcontext, localcontext
from pathlib import Path
from typing import Tuple, Literal
import sys
import time

# NOTE: Previously we only matched numbers that already had a decimal point, so
# plain integers (e.g. "5") were left unscaled while "5.0" became scaled. That
# produced mixed scaling in the fixed output. We now match ALL numbers (ints or
# floats) and apply the quantization multiplier uniformly so fixed-point values
# are consistent.

NUMBER_RE = re.compile(
    r"""
    (?P<num>
      (?<!\w)            # not preceded by word char
      [+-]?               # optional sign
      (?:
         \d+\.\d+       # digits.digits
        |\d+             # or just integer digits
        |\.\d+          # or leading-decimal like .5
      )
      (?!\w)             # not followed by word char
    )
    """,
    re.VERBOSE,
)

def count_decimal_places(s: str) -> int:
    if "." not in s:
        return 0
    s2 = s.lstrip("+-")
    if "." not in s2:
        return 0
    _, frac = s2.split(".", 1)
    frac = frac.rstrip("0")
    return len(frac)

def normalize_file(
    input_path_str: str,
    output_path_str: str = None,
    method: Literal['fast','legacy'] = 'fast',
    progress: bool = True,
    progress_every: int = 250000,
) -> Tuple[str, int]:
    """Convert *all* numeric probabilities (ints & floats) to a uniform fixed-point integer domain.

    Performance:
      - fast  : Two streaming passes (O(file size)), no huge in-memory string, no Decimal arithmetic.
                Suitable for multi-GB files. Only stores small rewrite buffers per line.
      - legacy: Original in-memory regex + Decimal approach (kept for reproducibility / validation).

    Returns (fixed_file_path, quant_mult).
    """
    in_path = Path(input_path_str)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Derive SF & output naming early
    stem = in_path.stem
    m_sf = re.search(r"_SF(\d+)_floating$", stem)
    if m_sf:
        sf_val = m_sf.group(1)
        base_stem = stem.replace(f"_SF{sf_val}_floating", f"_SF{sf_val}")
    else:
        sf_val = "1"
        base_stem = stem + "_SF1"
    if output_path_str:
        out_path = Path(output_path_str)
    else:
        out_path = in_path.with_name(f"{base_stem}_fixed{in_path.suffix or ''}")
    quant_sidecar = out_path.with_name(f"{base_stem}_quant_mult.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if method == 'legacy':
        text = in_path.read_text(encoding="utf-8")
        numbers = [m.group("num") for m in NUMBER_RE.finditer(text)]
        max_places = max((count_decimal_places(s) for s in numbers), default=0)
        quant_mult = 10 ** max_places
        if quant_mult != 1:
            getcontext().prec = 100
            with localcontext() as ctx:
                ctx.prec = 100
                scale_dec = Decimal(quant_mult)
                def repl(m: re.Match) -> str:
                    d = Decimal(m.group("num"))
                    scaled = d * scale_dec
                    return str(int(scaled.to_integral_value(rounding=ctx.rounding)))
                new_text = NUMBER_RE.sub(repl, text)
        else:
            new_text = text
        out_path.write_text(new_text, encoding='utf-8')
        quant_sidecar.write_text(str(quant_mult) + "\n", encoding='utf-8')
        print(f"[normalize] done method=legacy quant_mult={quant_mult} file={out_path}")
        return str(out_path), quant_mult

    # FAST streaming path
    t0 = time.time()
    max_places = 0
    total_numbers = 0
    total_lines = 0
    with in_path.open('r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            total_lines += 1
            for m in NUMBER_RE.finditer(line):
                total_numbers += 1
                places = count_decimal_places(m.group('num'))
                if places > max_places:
                    max_places = places
    quant_mult = 10 ** max_places

    if quant_mult == 1:
        shutil.copyfile(in_path, out_path)
        quant_sidecar.write_text("1\n", encoding='utf-8')
        print(f"[normalize] done method=fast quant_mult=1 file={out_path}")
        return str(out_path), 1

    # Precompute string of multiplier for optimization? (not needed) – implement scaling function.
    def scale_number_str(num_str: str) -> str:
        # Preserve sign
        sign = 1
        if num_str.startswith(('+','-')):
            if num_str[0] == '-':
                sign = -1
            num_body = num_str[1:]
        else:
            num_body = num_str
        if '.' in num_body:
            int_part, frac_part = num_body.split('.', 1)
        else:
            int_part, frac_part = num_body, ''
        # Remove leading empties (for forms like .5)
        if int_part == '':
            int_part = '0'
        # Pad frac_part to max_places (dataset ensures we won't exceed)
        frac_padded = frac_part.ljust(max_places, '0')
        scaled_val = sign * (int(int_part) * quant_mult + (int(frac_padded[:max_places]) if max_places else 0))
        return str(scaled_val)

    # Second pass write
    written_numbers = 0
    with in_path.open('r', encoding='utf-8', errors='replace') as src, out_path.open('w', encoding='utf-8') as dst:
        for line_idx, line in enumerate(src, start=1):
            # Replace numbers via manual scan (avoid building huge intermediate list)
            last_end = 0
            out_chunks = []
            for m in NUMBER_RE.finditer(line):
                out_chunks.append(line[last_end:m.start()])
                out_chunks.append(scale_number_str(m.group('num')))
                last_end = m.end()
                written_numbers += 1
            out_chunks.append(line[last_end:])
            dst.write(''.join(out_chunks))
            # Suppress incremental progress prints (kept minimal as requested)

    quant_sidecar.write_text(str(quant_mult) + "\n", encoding='utf-8')
    t1 = time.time()
    print(f"[normalize] done method=fast quant_mult={quant_mult} numbers={written_numbers} seconds={t1 - t0:.2f} file={out_path}")
    return str(out_path), quant_mult

def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Scale all decimal numbers in a file so they become integers."
    )
    parser.add_argument("input_file", help="Path to the input text file.")
    parser.add_argument("output_file", nargs='?', default=None, help="Optional. Path to the output file.")
    parser.add_argument("--method", choices=['fast','legacy'], default='fast', help="Normalization method (fast streaming or legacy Decimal).")
    parser.add_argument("--no-progress", action='store_true', help="Disable periodic progress output.")
    parser.add_argument("--progress-every", type=int, default=250000, help="Lines interval for progress messages (fast mode).")
    args = parser.parse_args()

    normalize_file(
        args.input_file,
        args.output_file,
        method=args.method,
        progress=not args.no_progress,
        progress_every=args.progress_every,
    )

if __name__ == "__main__":
    main()
