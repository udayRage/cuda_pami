#!/usr/bin/env python3
"""Generic metrics plotting utilities for experiment outputs.

Features:
- Accept a pandas DataFrame (or CSV path) plus column selections.
- Grouped multi-line plots: one line per group (e.g., algorithm), one PDF per y metric.
- Deterministic style assignment (color, marker, linestyle) with graceful cycling.
- Sorting of group lines by name or aggregate statistic for legend/readability.
- CLI interface for batch generation.

Example (library use):
    from scripts.plot_metrics import generate_metric_figures
    generate_metric_figures(df, dataset_name="Fuzzy_retail",
                            x_col="support_quant_int",
                            group_col="algorithm",
                            metrics=["exec_time","patterns_found"],
                            output_dir=Path("results/fuzzy/Fuzzy_retail/figures"))

CLI example:
    python -m scripts.plot_metrics metrics.csv \
        --x support_quant_int \
        --y exec_time cpu_mem_mb patterns_found \
        --group algorithm \
        --out results/fuzzy/Fuzzy_retail/figures \
        --width 5 --height 3

Return value (library): list of Path objects for generated PDFs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Sequence, Dict, Iterable, Tuple
import math
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Matplotlib config for LaTeX-friendly PDFs
plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
})

_DEFAULT_COLORS = [f'C{i}' for i in range(10)]  # tab10
_MARKERS = ['o','s','^','v','D','P','X','*','<','>','h','H','8','p','d']
_LINESTYLES = ['-','--','-.',':']

class StyleAssigner:
    """Assign a unique (color, marker, linestyle) triple per group, cycling deterministically."""
    def __init__(self,
                 colors: Sequence[str] | None = None,
                 markers: Sequence[str] | None = None,
                 linestyles: Sequence[str] | None = None):
        self.colors = list(colors) if colors else list(_DEFAULT_COLORS)
        self.markers = list(markers) if markers else list(_MARKERS)
        self.linestyles = list(linestyles) if linestyles else list(_LINESTYLES)

    def assign(self, groups: Sequence[str]) -> Dict[str, Tuple[str,str,str]]:
        styles: Dict[str, Tuple[str,str,str]] = {}
        total_colors = len(self.colors)
        total_markers = len(self.markers)
        total_ls = len(self.linestyles)
        for idx, g in enumerate(groups):
            c = self.colors[idx % total_colors]
            m = self.markers[(idx // total_colors) % total_markers]
            ls = self.linestyles[(idx // (total_colors * total_markers)) % total_ls]
            styles[g] = (c,m,ls)
        return styles

# Sorting strategies for group ordering / legend readability
_DEF_SORT = 'name'  # alternatives: mean, final, max

def _sort_groups(df: pd.DataFrame, group_col: str, y_col: str, strategy: str) -> List[str]:
    groups = df[group_col].unique().tolist()
    if strategy == 'name':
        return sorted(groups)
    agg = df.groupby(group_col)[y_col]
    if strategy == 'mean':
        order = agg.mean().sort_values()
    elif strategy == 'max':
        order = agg.max().sort_values()
    elif strategy == 'final':
        # final = y at the largest x per group (assuming x monotonic within group)
        # We'll approximate by taking last row per group in original order
        last_vals = df.sort_values(group_col).groupby(group_col)[y_col].tail(1)
        order = last_vals.sort_values()
    else:
        return sorted(groups)  # fallback
    return order.index.tolist()

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

_DEF_LABELS = {
    'exec_time': 'Execution Time (s)',
    'cpu_mem_mb': 'Peak CPU Memory (MB)',
    'gpu_mem_bytes': 'GPU Memory (Bytes)',
    'patterns_found': 'Patterns Found',
}

def plot_single_metric(df: pd.DataFrame,
                       x_col: str,
                       y_col: str,
                       group_col: str,
                       output_dir: Path,
                       dataset_name: str | None = None,
                       width: float = 5.0,
                       height: float = 3.0,
                       legend_loc: str = 'best',
                       tight: bool = True,
                       sort_groups: str = _DEF_SORT,
                       scale_x: bool = False,
                       quant_col: str = 'quant_mult') -> Path:
    d = df.copy()
    # Optional x scaling by quant multiplier
    if scale_x and quant_col in d.columns:
        d[x_col] = d[x_col] * d[quant_col]
    # Derived GPU memory MB convenience if requested y is gpu_mem_mb
    if y_col == 'gpu_mem_mb' and 'gpu_mem_mb' not in d.columns and 'gpu_mem_bytes' in d.columns:
        d['gpu_mem_mb'] = d['gpu_mem_bytes'] / (1024**2)
    if y_col not in d.columns:
        print(f"[plot] Skip missing y column: {y_col}")
        return output_dir / f"SKIPPED_{y_col}.txt"
    order = _sort_groups(d, group_col, y_col, sort_groups)
    styler = StyleAssigner()
    style_map = styler.assign(order)

    _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(width, height))
    for g in order:
        sub = d[d[group_col] == g].sort_values(x_col)
        if sub.empty:
            continue
        c,m,ls = style_map[g]
        ax.plot(sub[x_col], sub[y_col], label=g, color=c, marker=m, linestyle=ls)
    xlabel = x_col if not scale_x else f"{x_col} * {quant_col}"
    ylabel = _DEF_LABELS.get(y_col, y_col)
    title = f"{dataset_name} – {ylabel}" if dataset_name else ylabel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle=':')
    ax.legend(loc=legend_loc)
    if tight:
        fig.tight_layout()
    # Safe filename
    base = f"{dataset_name + '_' if dataset_name else ''}{y_col}.pdf"
    out_path = output_dir / base
    fig.savefig(out_path, format='pdf')
    plt.close(fig)
    print(f"[plot] Wrote {out_path}")
    return out_path


def generate_metric_figures(df: pd.DataFrame,
                            dataset_name: str,
                            x_col: str,
                            group_col: str,
                            metrics: Sequence[str],
                            output_dir: Path,
                            width: float = 5.0,
                            height: float = 3.0,
                            legend_loc: str = 'best',
                            tight: bool = True,
                            sort_groups: str = _DEF_SORT,
                            scale_x: bool = False) -> List[Path]:
    outputs: List[Path] = []
    for y_col in metrics:
        p = plot_single_metric(df, x_col, y_col, group_col, output_dir, dataset_name,
                               width=width, height=height, legend_loc=legend_loc, tight=tight,
                               sort_groups=sort_groups, scale_x=scale_x)
        outputs.append(p)
    return outputs

# ---------------- CLI -----------------

def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate line plots (PDF) from metrics CSV.")
    ap.add_argument('csv', help='Metrics CSV path')
    ap.add_argument('--x', required=True, help='X axis column')
    ap.add_argument('--y', nargs='+', required=True, help='One or more y metric columns')
    ap.add_argument('--group', required=True, help='Grouping column for separate lines')
    ap.add_argument('--dataset-name', default=None, help='Dataset name (for titles/file names)')
    ap.add_argument('--out', required=True, help='Output directory for PDFs')
    ap.add_argument('--width', type=float, default=5.0)
    ap.add_argument('--height', type=float, default=3.0)
    ap.add_argument('--legend-loc', default='best')
    ap.add_argument('--sort-groups', default=_DEF_SORT, choices=['name','mean','max','final'])
    ap.add_argument('--scale-x', action='store_true', help='Multiply x by quant_mult column if present')
    return ap.parse_args(argv)

def main(argv: Sequence[str] | None = None):
    ns = _parse_args(argv if argv is not None else sys.argv[1:])
    df = pd.read_csv(ns.csv)
    out_dir = Path(ns.out)
    generate_metric_figures(df, dataset_name=ns.dataset_name or 'dataset',
                            x_col=ns.x, group_col=ns.group, metrics=ns.y,
                            output_dir=out_dir, width=ns.width, height=ns.height,
                            legend_loc=ns.legend_loc, sort_groups=ns.sort_groups, scale_x=ns.scale_x)

if __name__ == '__main__':  # pragma: no cover
    main()
