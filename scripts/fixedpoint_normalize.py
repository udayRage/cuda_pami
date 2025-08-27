#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import re
import math
from typing import Tuple, Optional
from decimal import Decimal

try:
    import cudf
    import cupy as cp
    _CUDF_AVAILABLE = True
except ImportError:
    _CUDF_AVAILABLE = False
    raise SystemExit('cuDF not available; install RAPIDS to use this script.')
    
try:
    import pandas as pd
except ImportError:
    pass

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
    min_sup: Optional[float] = None,
) -> Tuple[str, str, str, int]:
    """Run full GPU pipeline.

    Returns (fixed_text_path, fixed_parquet_path, floating_parquet_path, quant_mult)
    """
    floating_path = Path(floating_path_str)
    if not floating_path.exists():
        raise FileNotFoundError(f"Input floating file not found: {floating_path}")
    base, fixed_text, fixed_parquet, floating_parquet, quant_file = _infer_sf_and_paths(floating_path)

    if (fixed_parquet.exists() and floating_parquet.exists() and quant_file.exists()) and not force:
        if write_fixed_text and not fixed_text.exists():
            print("Warning: Fixed text file is missing. Re-running the pipeline to generate it.")
        else:
            quant_mult = int(quant_file.read_text().strip())
            print(f"[gpu-pipeline] done reuse quant_mult={quant_mult} file={fixed_parquet}")
            return str(fixed_text), str(fixed_parquet), str(floating_parquet), quant_mult

    df = cudf.read_csv(
        floating_path,
        sep=":",
        names=["items", "values"],
        header=None,
        dtype=["str", "str"],
        skip_blank_lines=True,
    ).fillna("")
    
    ws_pat = r"[\t\r\n ]+$"
    df["items"] = df["items"].str.replace(ws_pat, "", regex=True)
    df["values"] = df["values"].str.replace(ws_pat, "", regex=True)

    # Generate 1-based transaction IDs; use cudf.RangeIndex for compatibility across RAPIDS versions
    df['txn_id'] = (cudf.Series(cudf.RangeIndex(start=0, stop=len(df))) + 1).astype("uint32")
    
    df_long = cudf.DataFrame({
        'item': df['items'].str.split('\t').explode(),
        'prob_str': df['values'].str.split('\t').explode(),
        'txn_id': df['txn_id'].repeat(df['items'].str.count('\t') + 1),
    }).dropna().reset_index(drop=True)
    
    df_long = df_long[df_long['item'] != ''].reset_index(drop=True)

    # Determine maximum number of decimal places present
    decimals_each = df_long["prob_str"].str.partition(".")[2].fillna("").str.len().fillna(0)
    max_sf = int(decimals_each.max())

    if min_sup is not None:
        raw_min_sup = Decimal(str(min_sup))
        if raw_min_sup.as_tuple().exponent < 0:
            max_sf = max(max_sf, -raw_min_sup.as_tuple().exponent)

    quant_mult = 10 ** max_sf

    parts = df_long["prob_str"].str.partition(".")
    int_part = parts[0].fillna("0")
    if max_sf > 0:
        frac_part_raw = parts[2].fillna("")
        # slice to max_sf then right-pad with zeros
        frac_scaled_str = frac_part_raw.str.slice(stop=max_sf).str.pad(width=max_sf, side='right', fillchar='0')
        # All entries are digits (possibly "0" * max_sf)
        scaled = int_part.astype("int64") * quant_mult + frac_scaled_str.astype("int64")
    else:
        scaled = int_part.astype("int64")

    df_long["prob"] = scaled.astype("uint64")
    
    if int(df_long['prob'].max()) >= 2**32:
        raise OverflowError("Scaled probabilities exceed uint32 range.")
    df_long['prob'] = df_long['prob'].astype('uint32')
    
    df_long['prob_float'] = df_long['prob_str'].astype('float64')
    if prefer_float32:
        float_out = df_long[['item', 'prob_float', 'txn_id']].rename(columns={'prob_float': 'prob'})
        float_out['prob'] = float_out['prob'].astype('float32')
    else:
        float_out = df_long[['item', 'prob_float', 'txn_id']].rename(columns={'prob_float': 'prob'})

    float_out.to_parquet(floating_parquet, compression='zstd')

    fixed_out = df_long[['item', 'prob', 'txn_id']]
    fixed_out.to_parquet(fixed_parquet, compression='zstd')

    if write_fixed_text:
        grouped = fixed_out.to_pandas().groupby('txn_id')
        with fixed_text.open('w', encoding='utf-8') as ft:
            for _, sub in grouped:
                items = sub['item'].tolist()
                probs = sub['prob'].tolist()
                ft.write('\t'.join(items) + ':' + '\t'.join(str(p) for p in probs) + '\n')
    
    quant_file.write_text(str(quant_mult) + '\n', encoding='utf-8')
    print(f"[gpu-pipeline] done rows={len(fixed_out)} quant_mult={quant_mult} fixed_parquet={fixed_parquet} floating_parquet={floating_parquet}")

    return str(fixed_text), str(fixed_parquet), str(floating_parquet), quant_mult


def normalize_file(input_path_str: str, output_path_str: str | None = None, progress: bool = True, write_fixed_text: bool = True):
    """Backward-compatible wrapper that produces fixed text + quant_mult.

    Parameters
    ----------
    input_path_str : str
        Path to floating transactional file (items	...:vals). Should include _floating in name for canonical naming.
    output_path_str : str | None
        Optional explicit fixed text output path (else pipeline naming used). If provided and differs, file is copied.
    progress : bool
        Ignored (kept for signature compatibility).
    write_fixed_text : bool
        If False, skips writing the text file and returns the parquet path instead.

    Returns
    -------
    (path, quant_mult)
        path is fixed_text_path if write_fixed_text is True, else fixed_parquet_path
    """
    fixed_txt, fixed_parq, float_parq, qm = gpu_fixedpoint_pipeline(input_path_str, write_fixed_text=write_fixed_text, force=False)
    if not write_fixed_text:
        return fixed_parq, qm

    if output_path_str and output_path_str != fixed_txt:
        # copy to user-requested location
        Path(output_path_str).write_text(Path(fixed_txt).read_text())
        fixed_txt = output_path_str
    return fixed_txt, qm


def main():
    ap = argparse.ArgumentParser(description="GPU fixed-point + parquet pipeline (cuDF)")
    ap.add_argument('floating_file', help='Path to *_floating.csv source')
    ap.add_argument('--no-fixed-text', action='store_true', help='Skip writing fixed text file (only parquets)')
    ap.add_argument('--force', action='store_true', help='Rebuild even if outputs exist')
    ap.add_argument('--float64', action='store_true', help='Keep floating parquet in float64 (default float32)')
    ap.add_argument('--min-sup', type=float, help='Minimum support value to consider for scaling factor.')
    args = ap.parse_args()
    
    gpu_fixedpoint_pipeline(
        args.floating_file,
        write_fixed_text=not args.no_fixed_text,
        force=args.force,
        prefer_float32=not args.float64,
        min_sup=args.min_sup,
    )

if __name__ == '__main__':
    main()