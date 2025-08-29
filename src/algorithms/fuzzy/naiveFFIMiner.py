#!/usr/bin/env python3
"""
naiveFFIMiner (STRICT DEVICE / NO-GDS)
=====================================

This variant keeps your original Polars→Arrow→cuDF pipeline but guarantees, for the
*duration of this miner only*:
  • **No Unified Memory** (no cudaMallocManaged) 
  • **No GDS** (KVIKIO compat mode forced ON → POSIX path)
  • **Allocator isolation** even if another module in the same process set RMM/CuPy to
    managed memory or GDS previously.

How it works
------------
- Wrap the whole `.mine()` run inside two context managers:
  (1) `EnvScope(KVIKIO_COMPAT_MODE=ON)` → disables cuFile (GDS) for any cuDF/KvikIO I/O
  (2) `AllocatorScope(device_only=True)` → installs a **device-only** pool
      for both cuDF (via RMM if available) and CuPy, then restores prior settings.
- We never call `cudf.read_parquet`; input parsing stays in **Polars**,
  so file I/O never touches GDS anyway. The env guard is belt-and-suspenders.

This file intentionally avoids any global allocator init at import-time.
"""
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Union, Dict, Tuple, Optional
import re
import numpy as np

# Keep Polars CPU-side import early (user wanted it)
os.environ.setdefault("POLARS_MAX_THREADS", str(os.cpu_count() or 1))
import polars as pl

# Base framework
from ..base import BaseAlgorithm  # type: ignore

# -----------------------------
# CUDA / host data structures
# -----------------------------
THREADS = 1024
INV_LOAD_FACTOR = 2
SENTINEL = 0
FIRST_ID = 1

KVPair = np.dtype([("line", np.uint32), ("probability", np.uint64)], align=True)
HashTable = np.dtype([
    ("table_ptr", np.uintp), ("table_size", np.uint32), ("mask", np.uint32), ("id", np.uint32)
], align=True)
assert KVPair.itemsize == 16
assert HashTable.itemsize == 24

# -----------------------------
# CUDA kernels (same as before)
# -----------------------------
KERNEL_SRC = r"""
// ------------------------------------------------------------
// Common helpers & data structures
// ------------------------------------------------------------
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
struct KVPair { uint32_t line; uint64_t probability; };
struct hash_table { KVPair* table; uint32_t length; uint32_t mask; uint32_t id; };
__device__ __forceinline__ uint32_t mul_hash(uint32_t k) { return k * 0x9E3779B1u; }
__device__ __forceinline__ uint64_t ullmin_dev(uint64_t a, uint64_t b) { return (a < b) ? a : b; }
#define SENTINEL 0u
#define FIRST_ID 1u
extern "C" __global__ void number_of_new_candidates_to_generate(const uint32_t *c, uint32_t n, uint32_t k, uint32_t *out) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; if (i >= n) return;
    if (k == 1) { out[i+1] = n - i - 1; return; }
    for (uint32_t j = i + 1; j < n; ++j) {
        bool same = true; for (uint32_t l = 0; l < k - 1; ++l) { if (c[i*k+l] != c[j*k+l]) { same = false; break; } }
        if (same) atomicAdd(&out[i+1], 1u);
    }
}
extern "C" __global__ void write_the_new_candidates(const uint32_t *c, uint32_t n, uint32_t k, const uint32_t *idx, uint32_t *out) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; if (i >= n) return;
    uint32_t slice = idx[i+1] - idx[i]; if (!slice) return;
    uint32_t index = idx[i] * (k + 1);
    for (uint32_t j = 0; j < slice; ++j) {
        for (uint32_t l = 0; l < k; ++l) { out[index + j*(k+1) + l] = c[i*k+l]; }
        out[index + j*(k+1) + k] = c[(i+1+j)*k + k-1];
    }
}
__device__ __forceinline__ void ht_insert(hash_table &ht, uint32_t key, uint64_t val) {
    uint32_t h = mul_hash(key) & ht.mask;
    while (true) {
        uint32_t old = atomicCAS(&ht.table[h].line, SENTINEL, key);
        if (old == SENTINEL || old == key) { atomicExch(&ht.table[h].probability, val); break; }
        h = (h + 1u) & ht.mask;
    }
}
extern "C" __global__ void build_tables(uint64_t total, const uint32_t *items, const uint32_t *lines, const uint64_t *probs, hash_table* hts) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x; if (tid >= total) return;
    ht_insert(hts[items[tid]], lines[tid], probs[tid]);
}
__device__ uint64_t ht_get(const hash_table &ht, uint32_t key) {
    uint32_t h = mul_hash(key) & ht.mask;
    while (true) {
        uint32_t cur = ht.table[h].line; if (cur == key) return ht.table[h].probability;
        if (cur == SENTINEL) return 0u; h = (h + 1u) & ht.mask;
    }
}
extern "C" __global__ void mine_kernel(const hash_table* hts, const uint32_t* cands, uint32_t nCands, uint32_t candSize, uint64_t* supports) {
    uint32_t bid = blockIdx.x; if (bid >= nCands) return;
    const hash_table& htEnd = hts[cands[bid*candSize + candSize - 1]];
    uint32_t reqHits = candSize - 1u; uint64_t local_support = 0;
    for (uint32_t i = threadIdx.x; i < htEnd.length; i += blockDim.x) {
        uint32_t line = htEnd.table[i].line; if (line == SENTINEL) continue;
        uint64_t min_p = htEnd.table[i].probability; uint32_t hits = 0u;
        for (uint32_t j = 0; j < reqHits; ++j) {
            uint64_t p = ht_get(hts[cands[bid*candSize + j]], line);
            if (p == 0) { hits = 0u; break; } min_p = ullmin_dev(min_p, p); ++hits;
        }
        if (hits == reqHits) local_support += min_p;
    }
    if (local_support != 0) atomicAdd(&supports[bid], local_support);
}
"""

# -------------------------------------------
# Small context managers to isolate settings
# -------------------------------------------
class EnvScope:
    def __init__(self, **env_updates: str):
        self.env_updates = env_updates
        self.prev: Dict[str, Optional[str]] = {}
    def __enter__(self):
        for k, v in self.env_updates.items():
            self.prev[k] = os.environ.get(k)
            os.environ[k] = v
        return self
    def __exit__(self, exc_type, exc, tb):
        for k, prev_v in self.prev.items():
            if prev_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev_v

class AllocatorScope:
    """Force device-only pools for both cuDF(RMM) and CuPy, then restore previous.
    If RMM isn't available, fall back to a CuPy device-only pool.
    """
    def __init__(self):
        self._had_rmm = False
        self._prev_rmm = None
        self._prev_cp_alloc = None
        self._prev_cp_pinned = None
        self._cp_pool = None
    def __enter__(self):
        import cupy as cp
        # Save current CuPy allocators
        self._prev_cp_alloc = cp.cuda.get_allocator()
        try:
            self._prev_cp_pinned = cp.cuda.get_pinned_memory_allocator()
        except Exception:
            self._prev_cp_pinned = None
        # Try RMM first
        try:
            import rmm
            from rmm.allocators.cupy import rmm_cupy_allocator
            self._had_rmm = True
            # Save previous RMM MR
            try:
                self._prev_rmm = rmm.mr.get_current_device_resource()
            except Exception:
                self._prev_rmm = None
            # Install a device-only pool over cudaMalloc
            mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource())
            rmm.mr.set_current_device_resource(mr)
            cp.cuda.set_allocator(rmm_cupy_allocator)
            # Ensure no pinned pool (keep it simple/strict)
            cp.cuda.set_pinned_memory_allocator(None)
            return self
        except Exception:
            # Fall back: CuPy-only device pool
            self._cp_pool = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(self._cp_pool.malloc)
            cp.cuda.set_pinned_memory_allocator(None)
            return self
    def __exit__(self, exc_type, exc, tb):
        import cupy as cp
        # Restore RMM MR if we changed it
        if self._had_rmm:
            try:
                import rmm
                if self._prev_rmm is not None:
                    rmm.mr.set_current_device_resource(self._prev_rmm)
            except Exception:
                pass
        # Restore CuPy allocators
        try:
            cp.cuda.set_allocator(self._prev_cp_alloc)
        except Exception:
            pass
        try:
            cp.cuda.set_pinned_memory_allocator(self._prev_cp_pinned)
        except Exception:
            pass
        # Free any CuPy pool we created
        try:
            if self._cp_pool is not None:
                self._cp_pool.free_all_blocks()
        except Exception:
            pass

# -----------------------------
# Miner implementation
# -----------------------------
class naiveFFIMiner(BaseAlgorithm):
    """Advanced (Polars + CUDA) miner with strict device memory + no GDS.

    Accepts floating fuzzy values and auto-detects a scaling factor; never uses
    managed memory or GDS during its run, regardless of global settings changed
    elsewhere in the same process.
    """
    def __init__(self, iFile: Union[str, Path], min_support: float, sep: str = "\t", quant_mult: int | None = None, debug: bool = False):
        super().__init__(iFile=iFile, debug=debug)
        self.min_support_float = float(min_support)
        self.forced_quant_mult = int(quant_mult) if (quant_mult is not None and quant_mult > 0) else None
        self._sep = sep
        self._gpu_memory_usage = 0
        self.patterns: Dict[Tuple[str, ...], float] = {}

        self._gpu_memory_usage = 0
        self._gpu_mem_peak = 0  # can keep or drop; superseded by _peak_driver_used

        # New telemetry fields (mirrors cuFFIMiner)
        self._peak_driver_used = 0
        self._peak_pool_used = 0
        self._peak_pool_total = 0
        self._rmm_peak_bytes = None
        try:
            import rmm.statistics as rstats
            rstats.enable_statistics()
            self._rmm_stats = rstats
        except Exception:
            self._rmm_stats = None

    def _snap_mem(self) -> None:
        """Update peak snapshots for driver & pool usage."""
        try:
            import cupy as cp
            cp.cuda.runtime.deviceSynchronize()
            try:
                free_b, total_b = cp.cuda.runtime.memGetInfo()
                used_b = int(total_b) - int(free_b)
                if used_b > self._peak_driver_used:
                    self._peak_driver_used = used_b
            except Exception:
                pass
            try:
                pool = cp.get_default_memory_pool()
                if pool is not None:
                    self._peak_pool_used = max(self._peak_pool_used, pool.used_bytes())
                    self._peak_pool_total = max(self._peak_pool_total, pool.total_bytes())
            except Exception:
                pass
        except Exception:
            # CUDA not ready yet / allocator not installed yet
            pass


    # ---------- CPU load & scale (Polars) ----------
    def _cpu_load_and_scale_data(self):
        path = str(self._iFile)
        is_parquet = path.lower().endswith('.parquet')
        if is_parquet:
            df = pl.read_parquet(path)
            value_col = 'prob'
        else:
            lazy = (
                pl.scan_csv(path, has_header=False, separator=":", new_columns=["items_str", "values_str"])\
                  .with_columns(pl.col("items_str").str.split(self._sep), pl.col("values_str").str.split(self._sep))\
                  .filter(pl.col("items_str").list.len() == pl.col("values_str").list.len())\
                  .explode(["items_str", "values_str"]).rename({"items_str": "item", "values_str": "prob"})
            )
            df = lazy.with_row_index("txn_id", offset=1).collect()
            value_col = 'prob'
        # Determine scale
        if self.forced_quant_mult is not None:
            self.scale_factor = self.forced_quant_mult
        else:
            max_decimals = (
                df.get_column(value_col).cast(pl.Utf8).str.split_exact('.', 1)
                  .struct.field('field_1').str.len_chars().fill_null(0).max()
            )
            if max_decimals is None:
                max_decimals = 0
            self.scale_factor = 10 ** int(max_decimals)
        self._minSup_scaled = int(self.min_support_float * self.scale_factor)
        # Scale and select
        df = df.with_columns((pl.col(value_col).cast(pl.Float64) * self.scale_factor).cast(pl.UInt32).alias('prob_scaled'))\
               .select(pl.col('item').cast(pl.Utf8), pl.col('prob_scaled').alias('prob'), pl.col('txn_id').cast(pl.UInt32))
        # Hand off to cuDF via Arrow AFTER allocator scope is active
        return df

    # ---------- GPU-side helpers ----------
    @staticmethod
    def _to_device_struct(arr: np.ndarray):
        import cupy as cp
        return cp.asarray(arr.view(np.uint8)).view(arr.dtype)

    def _calculate_support_and_filter_items(self, gdf):
        import cupy as cp
        support_df = gdf.groupby("item").agg({"prob": "sum", "txn_id": "count"}).rename(columns={"txn_id": "freq"})
        support_df = support_df[support_df["prob"] >= self._minSup_scaled].sort_values("freq", ascending=False).reset_index()
        if len(support_df) == 0:
            return None, None
        n_items = len(support_df)
        support_df["new_id"] = cp.arange(FIRST_ID, FIRST_ID + n_items, dtype=cp.uint32)
        self.max_id = int(support_df["new_id"].max())
        self.rename_map = np.empty(self.max_id + 1, dtype=object)
        self.rename_map[support_df["new_id"].to_numpy()] = support_df["item"].to_numpy()
        gdf = gdf[gdf["item"].isin(support_df["item"])]
        gdf = gdf.merge(support_df[["item", "new_id"]], on="item")
        gdf = gdf.drop(columns=["item"]).rename(columns={"new_id": "item"})
        self.support_df = support_df
        return support_df, gdf

    def _allocate_gpu_hash_tables(self, support_df):
        import cupy as cp
        freq_cp = support_df["freq"].to_cupy(dtype=cp.uint32)
        needed_size = (freq_cp * INV_LOAD_FACTOR).astype(cp.float32)
        sizes_cp = cp.power(2, cp.ceil(cp.log2(needed_size))).astype(cp.uint32)
        slots_total = int(sizes_cp.sum())
        self.buckets = cp.zeros(slots_total, dtype=KVPair)
        base_ptr = self.buckets.data.ptr
        offset = cp.cumsum(cp.concatenate([cp.array([0], dtype=cp.uint64), sizes_cp.astype(cp.uint64)])) * KVPair.itemsize
        ht_host = np.zeros(self.max_id + 1, dtype=HashTable)
        sizes_host = sizes_cp.get()
        offset_host = offset.get()
        for idx, id_ in enumerate(support_df["new_id"].to_numpy()):
            ht_host[id_]["table_ptr"] = base_ptr + offset_host[idx]
            ht_host[id_]["table_size"] = sizes_host[idx]
            ht_host[id_]["mask"] = sizes_host[idx] - 1
            ht_host[id_]["id"] = id_
        self._max_threads = min(int(sizes_cp.min()), THREADS) if len(sizes_cp) > 0 else THREADS
        return ht_host

    def _build_gpu_hash_tables(self, ht_host, gdf):
        import cupy as cp
        self.item_col = gdf["item"].astype("uint32").values
        self.line_col = gdf["txn_id"].astype("uint32").values
        self.prob_col = gdf["prob"].astype("uint64").values
        self._ht_dev = self._to_device_struct(ht_host)
        total_pairs = len(self.item_col)
        grid_size = (total_pairs + self._max_threads - 1) // self._max_threads
        self.KERN_BUILD((grid_size,), (self._max_threads,), (total_pairs, self.item_col, self.line_col, self.prob_col, self._ht_dev))
        cp.cuda.runtime.deviceSynchronize()

    def _generate_next_candidates(self, cands, k: int):
        import cupy as cp
        n = len(cands)
        if n == 0:
            return cp.array([], dtype=cp.uint32)
        grid_size = (n + self._max_threads - 1) // self._max_threads
        num = cp.zeros(n + 1, dtype=cp.uint32)
        self.KERN_NUM((grid_size,), (self._max_threads,), (cands, n, k, num))
        idx = cp.cumsum(num, dtype=cp.uint32)
        out_len = int(idx[-1].get())
        if out_len == 0:
            return cp.array([], dtype=cp.uint32)
        nxt = cp.zeros((out_len, k + 1), dtype=cp.uint32)
        self.KERN_WRITE((grid_size,), (self._max_threads,), (cands, n, k, idx, nxt))
        return nxt

    def _compile_kernels(self):
        import cupy as cp
        NVCC_OPTIONS = (
            "-std=c++14",
            "-O3",
            "-Xptxas=-v",
            "-Xptxas=--warn-on-spills",
            "-Xptxas=-dlcm=ca",
            "-lineinfo",
        )
        mod = cp.RawModule(code=KERNEL_SRC, backend="nvcc", options=NVCC_OPTIONS)
        self.KERN_NUM   = mod.get_function("number_of_new_candidates_to_generate")
        self.KERN_WRITE = mod.get_function("write_the_new_candidates")
        self.KERN_BUILD = mod.get_function("build_tables")
        self.KERN_MINE  = mod.get_function("mine_kernel")

    # ---------- Orchestrator ----------
    def _mine(self):
        # Guard env and allocators: NO GDS + device-only pools
        with EnvScope(KVIKIO_COMPAT_MODE="ON"), AllocatorScope():
            import cupy as cp
            import cudf
            if not cp.cuda.is_available():
                raise RuntimeError("CUDA is not available")

            # After allocator install
            self._snap_mem()

            # Compile kernels
            self._compile_kernels()
            self._snap_mem()

            # CPU load & scale → cuDF (device-side)
            df_polars = self._cpu_load_and_scale_data()
            exploded_gdf = cudf.DataFrame.from_arrow(df_polars.to_arrow())
            self._snap_mem()

            # Filter & build
            support_df, final_gdf = self._calculate_support_and_filter_items(exploded_gdf)
            if support_df is None:
                self.patterns = {}
                # Optionally still print results (see print_results override)
                return

            ht_host = self._allocate_gpu_hash_tables(support_df)
            self._snap_mem()

            self._build_gpu_hash_tables(ht_host, final_gdf)
            cp.cuda.runtime.deviceSynchronize()
            self._snap_mem()

            # Mining loop
            k = 1
            candidates = self.support_df["new_id"].to_cupy().reshape(-1, 1)
            gpu_results = []
            while len(candidates) > 0:
                next_candidates = self._generate_next_candidates(candidates, k)
                self._snap_mem()
                if len(next_candidates) == 0:
                    break
                k += 1
                supports = cp.zeros(len(next_candidates), dtype=cp.uint64)
                self._snap_mem()
                self.KERN_MINE((len(next_candidates),), (self._max_threads,),
                            (self._ht_dev, next_candidates, len(next_candidates), k, supports))
                cp.cuda.runtime.deviceSynchronize()
                self._snap_mem()

                mask = supports >= self._minSup_scaled
                frequent_cands = next_candidates[mask]
                if len(frequent_cands) > 0:
                    gpu_results.append((frequent_cands.get(), supports[mask].get()))
                candidates = frequent_cands

            # Results
            self._process_results(gpu_results)

            # Optional RMM stats peak (if available)
            if self._rmm_stats is not None:
                try:
                    stats = self._rmm_stats.get_statistics()
                    self._rmm_peak_bytes = getattr(stats, "peak_bytes", None)
                except Exception:
                    pass

            # Record final driver peak for external access
            self._gpu_memory_usage = self._peak_driver_used

            # Try to free CuPy pool blocks (AllocatorScope will also restore allocators)
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass


    def _process_results(self, gpu_results):
        patterns = {
            (item,): float(prob) / float(self.scale_factor)
            for item, prob in zip(self.support_df["item"].to_numpy(), self.support_df["prob"].to_numpy())
        }
        for ids_cpu, vals in gpu_results:
            names = self.rename_map[ids_cpu]
            for name_tuple, sup in zip(names, vals):
                patterns[tuple(name_tuple)] = float(sup) / float(self.scale_factor)
        self.patterns = patterns

    def save(self, oFile: Union[str, Path]):
        with open(oFile, "w") as f:
            for itemset, support in self.patterns.items():
                f.write(f"{','.join(itemset)}\t{support}\n")

    def print_results(self):
        print(f"\n--- {self.__class__.__name__} Results ---")
        print(f"Execution Time: {self.get_execution_time():.4f} seconds")
        print(f"Peak CPU Memory Usage: {self.get_memory_usage():.2f} MB")
        print(f"Peak GPU (driver) Used: {self._peak_driver_used / (1024**2):.2f} MB")
        print(f"Peak Pool Used:        {self._peak_pool_used / (1024**2):.2f} MB")
        print(f"Peak Pool Total:       {self._peak_pool_total / (1024**2):.2f} MB")
        if self._rmm_peak_bytes is not None:
            print(f"RMM Statistics Peak:   {self._rmm_peak_bytes / (1024**2):.2f} MB")
        print(f"Patterns Found: {self.get_pattern_count()}")
        print("--------------------" + "-" * len(self.__class__.__name__))


    # ---------- CLI ----------
    @staticmethod
    def _cli():
        p = argparse.ArgumentParser(description="naiveFFIMiner (strict device / no GDS)")
        p.add_argument("iFile", type=str, help="Input dataset (.txt/.csv/.tsv or .parquet)")
        p.add_argument("min_support", type=float, help="Minimum support as float (e.g., 0.05)")
        p.add_argument("--quant-mult", type=int, default=None, help="Force quantization multiplier (bypass auto)")
        p.add_argument("-o", "--oFile", type=str, default="patterns.txt")
        p.add_argument("--sep", type=str, default="\t")
        p.add_argument("--debug", action="store_true")
        args = p.parse_args()
        algo = naiveFFIMiner(iFile=args.iFile, min_support=args.min_support, sep=args.sep, quant_mult=args.quant_mult, debug=args.debug)
        algo.mine(); algo.save(args.oFile); algo.print_results()

if __name__ == '__main__':
    naiveFFIMiner._cli()
