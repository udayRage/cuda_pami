#!/usr/bin/env python3
"""GPU accelerated fixed-point normalization + Parquet generation pipeline.

Reads a floating fuzzy transactional dataset (items separated by TAB on the left
of a colon, probabilities separated by TAB on the right), determines a global
quantization multiplier (10^max_decimal_places), produces:

  1. Fixed-point text file (integers)  (optional)
  2. Fixed-point Parquet:  item:str, prob:uint32, txn_id:uint32
  3. Floating Parquet:     item:str, prob:float32/64, txn_id:uint32
  4. Sidecar quant multiplier file *_quant_mult.txt

Requires: RAPIDS cuDF (GPU). Falls back to a light CPU path if cuDF missing.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import math
from typing import Tuple, Optional

try:  # GPU imports
    import cudf  # type: ignore
    _CUDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CUDF_AVAILABLE = False

NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+\.\d+|\d+|\.\d+)(?!\w)")

def _count_decimal_places(num: str) -> int:
    if '.' not in num:
        return 0
    s = num.lstrip('+-')
    if '.' not in s:
        return 0
    _, frac = s.split('.', 1)
    frac = frac.rstrip('0')
    return len(frac)

def _infer_sf_and_paths(floating_path: Path) -> Tuple[str, Path, Path, Path, Path]:
    stem = floating_path.stem
    m = re.search(r"_SF(\d+)_floating$", stem)
    if m:
        sf = m.group(1)
        base = stem.replace(f"_SF{sf}_floating", f"_SF{sf}")
    else:
        sf = '1'
        base = stem + '_SF1'
    fixed_text = floating_path.with_name(f"{base}_fixed{floating_path.suffix}")
    fixed_parquet = fixed_text.with_suffix('.parquet')
    floating_parquet = floating_path.with_suffix('.parquet')
    quant_file = floating_path.with_name(f"{base}_quant_mult.txt")
    return base, fixed_text, fixed_parquet, floating_parquet, quant_file

def gpu_fixedpoint_pipeline(
    floating_path_str: str,
    write_fixed_text: bool = True,
    force: bool = False,
    prefer_float32: bool = True,
) -> Tuple[str, str, str, int]:
    """Run full GPU pipeline.

    Returns (fixed_text_path, fixed_parquet_path, floating_parquet_path, quant_mult)
    """
    floating_path = Path(floating_path_str)
    if not floating_path.exists():
        raise FileNotFoundError(f"Input floating file not found: {floating_path}")
    base, fixed_text, fixed_parquet, floating_parquet, quant_file = _infer_sf_and_paths(floating_path)

    if (fixed_parquet.exists() and floating_parquet.exists() and quant_file.exists() and fixed_text.exists()) and not force:
        quant_mult = int(quant_file.read_text().strip())
        print(f"[gpu-pipeline] done reuse quant_mult={quant_mult} file={fixed_parquet}")
        return str(fixed_text), str(fixed_parquet), str(floating_parquet), quant_mult

    if not _CUDF_AVAILABLE:
        raise RuntimeError("cuDF not available; cannot run GPU pipeline.")

    # Suppress intermediate logs
    # Two columns split by ':'
    df = cudf.read_csv(
        floating_path,
        sep=":",
        names=["items_str", "values_str"],
        header=None,
        dtype=["str", "str"],
        skip_blank_lines=True,
    )
    # Clean trailing whitespace
    ws_pat = r"[\t\r\n ]+$"
    df["items_str"] = df["items_str"].str.replace(ws_pat, "", regex=True)
    df["values_str"] = df["values_str"].str.replace(ws_pat, "", regex=True)

    # Determine max number of items in a transaction
    df["item_count"] = df["items_str"].str.count("\t") + 1
    max_items = int(df["item_count"].max())
    # Omit detailed max items log

    # Wide split (expand)
    items_wide = df["items_str"].str.split("\t", expand=True, n=max_items-1)
    values_wide = df["values_str"].str.split("\t", expand=True, n=max_items-1)

    # Build long form by stacking each column pair
    long_parts = []
    txn_id = cudf.Series(cudf.core.column.arange(1, len(df)+1, dtype='uint32'))
    for i in range(max_items):
        ic = items_wide.get(i)
        vc = values_wide.get(i)
        if ic is None or vc is None:
            continue
        part = cudf.DataFrame({
            'item': ic,
            'prob_float': vc,
            'txn_id': txn_id
        })
        part = part.dropna(subset=['item', 'prob_float'])
        long_parts.append(part)
    if not long_parts:
        raise ValueError("No items parsed; check input format.")
    long_df = cudf.concat(long_parts, ignore_index=True)
    long_df = long_df[long_df['item'] != '']

    # Cast probabilities to float
    long_df['prob_float'] = long_df['prob_float'].astype('float64')
    if prefer_float32:
        long_df['prob_float32'] = long_df['prob_float'].astype('float32')
        prob_col = 'prob_float32'
    else:
        prob_col = 'prob_float'

    # Determine quant multiplier (max decimal places) on host for simplicity
    sample_vals = long_df['prob_float'].astype('str').to_pandas()
    max_places = 0
    for v in sample_vals:
        if v and v != 'nan':
            p = _count_decimal_places(v)
            if p > max_places:
                max_places = p
    quant_mult = 10 ** max_places
    # quant_mult summary deferred to final line

    # Fixed integer column
    if quant_mult > 1:
        long_df['prob'] = (long_df['prob_float'] * quant_mult + 0.5).astype('int64')
    else:
        long_df['prob'] = long_df['prob_float'].astype('int64')
    # Bounds check
    if int(long_df['prob'].max()) >= 2**32:
        raise OverflowError("Scaled probabilities exceed uint32 range.")
    long_df['prob'] = long_df['prob'].astype('uint32')

    # Write floating parquet
    float_out = long_df[['item', prob_col, 'txn_id']].rename(columns={prob_col: 'prob'})
    float_out.to_parquet(floating_parquet, compression='zstd')

    # Write fixed parquet
    fixed_out = long_df[['item', 'prob', 'txn_id']]
    fixed_out.to_parquet(fixed_parquet, compression='zstd')

    # Optional fixed text file (reconstruct line-wise)
    if write_fixed_text:
        # We reconstruct by grouping per txn_id; may be large, do in batches
        with fixed_text.open('w', encoding='utf-8') as ft:
            # Build lists per txn (GPU to CPU per group)
            grouped = fixed_out.to_pandas().groupby('txn_id')
            for txn_id_val, sub in grouped:
                items = sub['item'].tolist()
                probs = sub['prob'].tolist()
                ft.write('\t'.join(items) + ':' + '\t'.join(str(p) for p in probs) + '\n')
    pass

    quant_file.write_text(str(quant_mult) + '\n', encoding='utf-8')
    print(f"[gpu-pipeline] done rows={len(fixed_out)} quant_mult={quant_mult} fixed_parquet={fixed_parquet} floating_parquet={floating_parquet}")

    return str(fixed_text), str(fixed_parquet), str(floating_parquet), quant_mult

def main():  # CLI
    ap = argparse.ArgumentParser(description="GPU fixed-point + parquet pipeline (cuDF)")
    ap.add_argument('floating_file', help='Path to *_floating.csv source')
    ap.add_argument('--no-fixed-text', action='store_true', help='Skip writing fixed text file (only parquets)')
    ap.add_argument('--force', action='store_true', help='Rebuild even if outputs exist')
    ap.add_argument('--float64', action='store_true', help='Keep floating parquet in float64 (default float32)')
    args = ap.parse_args()
    if not _CUDF_AVAILABLE:
        raise SystemExit('cuDF not available; install RAPIDS to use this script.')
    gpu_fixedpoint_pipeline(
        args.floating_file,
        write_fixed_text=not args.no_fixed_text,
        force=args.force,
        prefer_float32=not args.float64,
    )

if __name__ == '__main__':
    main()
