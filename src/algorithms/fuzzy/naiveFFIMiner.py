"""
NaiveFFIMiner — CUDA Fuzzy Frequent Itemset Miner

Highlights
----------
• Switchable allocator (RMM device/managed or CuPy device/managed) + optional pinned host pool
• Manual, deterministic memory accounting per phase (no NVML / driver introspection)
• Public getters for “theory” memory: get_theory_memory_bytes(), get_theory_memory_report()
• Polars-based CPU load/explode/quantize; compact per-item dense arrays on GPU
• Single RawModule compile; safe candidate generation + mining kernels

Input assumes probabilities are float-like and get quantized to UInt32 via a scale factor.
"""
from __future__ import annotations

import os
import gc
import argparse
import copy
from pathlib import Path
from typing import Union, Dict, Tuple, Optional, Any, List

import time
import numpy as np
import cupy as cp
import polars as pl

# Use all CPU cores for Polars by default
os.environ.setdefault("POLARS_MAX_THREADS", str(os.cpu_count() or 1))

from ..base import BaseAlgorithm  # type: ignore


# -----------------------
# Global knobs & dtypes
# -----------------------
THREADS = 1024
INV_LOAD_FACTOR = 3/2
SENTINEL = 0
FIRST_ID = 1

KVPair = np.dtype([("line", np.uint32), ("probability", np.uint64)], align=True)
HashTable = np.dtype(
    [
        ("table_ptr", np.uintp),
        ("table_size", np.uint32),
        ("mask", np.uint32),
        ("id", np.uint32),
        ("array_ptr", np.uintp),
        ("array_size", np.uint32),
    ],
    align=True,
)


# -----------------------
# CUDA kernel source
# -----------------------
KERNEL_SRC = r"""
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

struct KVPair { uint32_t line; uint64_t probability; };

struct hash_table {
    KVPair*  table;   // open-addressing buckets
    uint32_t length;
    uint32_t mask;
    uint32_t id;
    KVPair*  arr;     // dense, compact array
    uint32_t arr_len;
};

__device__ __forceinline__ uint32_t mul_hash(uint32_t k){ return k * 0x9E3779B1u; }
__device__ __forceinline__ uint64_t ullmin_dev(uint64_t a, uint64_t b){ return (a < b) ? a : b; }
#define SENTINEL 0u
#define FIRST_ID 1u

extern "C" __global__ void number_of_new_candidates_to_generate(const uint32_t *c, uint32_t n, uint32_t k, uint32_t *out){
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; if (i >= n) return;
    if (k == 1) { out[i+1] = n - i - 1; return; }

    uint32_t count = 0;
    for (uint32_t j = i + 1; j < n; ++j){
        bool same = true;
        for (uint32_t l = 0; l < k - 1; ++l){
            if (c[i*k+l] != c[j*k+l]) { same = false; break; }
        }
        // if (same) atomicAdd(&out[i+1], 1u);
        if (same) count++;
    }

    out[i+1] = count;
}


extern "C" __global__ void write_the_new_candidates(const uint32_t *c, uint32_t n, uint32_t k, const uint32_t *idx, uint32_t *out){
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; if (i >= n) return;
    uint32_t slice = idx[i+1] - idx[i]; if (!slice) return;
    uint32_t base = idx[i] * (k + 1);
    for (uint32_t j = 0; j < slice; ++j){
        for (uint32_t l = 0; l < k; ++l){ out[base + j*(k+1) + l] = c[i*k+l]; }
        out[base + j*(k+1) + k] = c[(i+1+j)*k + (k-1)];
    }
}


extern "C" __global__ void write_dense_arrays(
    uint64_t total,
    const uint32_t* __restrict__ items,
    const uint32_t* __restrict__ lines,
    const uint64_t* __restrict__ probs,
    hash_table* __restrict__ hts,
    uint32_t* __restrict__ counters
){
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x; if (tid >= total) return;
    uint32_t it = items[tid];
    uint32_t pos = atomicAdd(&counters[it], 1u);
    if (pos < hts[it].arr_len){
        hts[it].arr[pos].line        = lines[tid];
        hts[it].arr[pos].probability = probs[tid];
    }
}

__device__ __forceinline__ void ht_insert(hash_table &ht, uint32_t key, uint64_t val){
    uint32_t h = mul_hash(key) & ht.mask;
    while (true){
        uint32_t old = atomicCAS(&ht.table[h].line, SENTINEL, key);
        if (old == SENTINEL || old == key){ atomicExch(&ht.table[h].probability, val); break; }
        h = (h + 1u) & ht.mask;
    }
}

extern "C" __global__ void build_tables(uint64_t total, const uint32_t *items, const uint32_t *lines, const uint64_t *probs, hash_table* hts){
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x; if (tid >= total) return;
    ht_insert(hts[items[tid]], lines[tid], probs[tid]);
}

__device__ uint64_t ht_get(const hash_table &ht, uint32_t key){
    uint32_t h = mul_hash(key) & ht.mask;
    while (true){
        uint32_t cur = ht.table[h].line; if (cur == key) return ht.table[h].probability;
        if (cur == SENTINEL) return 0u;
        h = (h + 1u) & ht.mask;
    }
}

extern "C" __global__ void mine_kernel(
    const hash_table* __restrict__ hts,
    const uint32_t*   __restrict__ cands,
    uint32_t nCands, uint32_t candSize,
    uint64_t* __restrict__ supports
){
    uint32_t bid = blockIdx.x; if (bid >= nCands) return;
    const hash_table& htEnd = hts[cands[bid*candSize + (candSize - 1)]];
    uint32_t reqHits = candSize - 1u;
    uint64_t local_support = 0;

    for (uint32_t i = threadIdx.x; i < htEnd.arr_len; i += blockDim.x){
        uint32_t line = htEnd.arr[i].line;
        uint64_t min_p = htEnd.arr[i].probability;
        uint32_t hits = 0u;
        #pragma unroll 1
        for (uint32_t j = 0; j < reqHits; ++j){
            uint64_t p = ht_get(hts[cands[bid*candSize + j]], line);
            if (p == 0){ hits = 0u; break; }
            min_p = ullmin_dev(min_p, p);
            ++hits;
        }
        if (hits == reqHits){
            local_support += min_p;
        }
    }
    if (local_support != 0){
        atomicAdd(&supports[bid], local_support);
    }
}
"""


# -----------------------
# Allocator & compilation
# -----------------------
def _init_allocator(mode: str, use_pinned: bool = False) -> Tuple[Optional[object], Optional[object]]:
    """Initialize a single device allocator (RMM or CuPy) and optional pinned host pool."""
    dev_pool = pin_pool = None
    m = mode.lower()
    if m.startswith("rmm"):
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator
        if m == "rmm_device":
            mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource())
        elif m == "rmm_managed":
            mr = rmm.mr.PoolMemoryResource(rmm.mr.ManagedMemoryResource())
        else:
            raise ValueError(f"Unknown allocator mode: {mode}")
        rmm.mr.set_current_device_resource(mr)
        cp.cuda.set_allocator(rmm_cupy_allocator)
        dev_pool = mr
    elif m == "cupy_device":
        dev_pool = cp.cuda.MemoryPool()
        cp.cuda.set_allocator(dev_pool.malloc)
    elif m == "cupy_managed":
        dev_pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
        cp.cuda.set_allocator(dev_pool.malloc)
    else:
        raise ValueError(f"Unknown allocator mode: {mode}")

    if use_pinned:
        pin_pool = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(pin_pool.malloc)
    return dev_pool, pin_pool


def _compile_kernels() -> Tuple[cp.RawKernel, cp.RawKernel, cp.RawKernel, cp.RawKernel, cp.RawKernel]:
    """Compile and return CUDA kernels from the embedded source with nvcc backend."""
    opts = ("-std=c++14", "-O3", "-Xptxas=-v", "-Xptxas=--warn-on-spills", "-Xptxas=-dlcm=ca", "-lineinfo")
    mod = cp.RawModule(code=KERNEL_SRC, backend="nvcc", options=opts)
    return (
        mod.get_function("number_of_new_candidates_to_generate"),
        mod.get_function("write_the_new_candidates"),
        mod.get_function("build_tables"),
        mod.get_function("mine_kernel"),
        mod.get_function("write_dense_arrays"),
    )


# -----------------------
# Miner implementation
# -----------------------
class naiveFFIMiner(BaseAlgorithm):
    """GPU-accelerated fuzzy frequent itemset miner with Polars preprocessing and per-item hash tables."""

    def __init__(
        self,
        iFile: Union[str, Path],
        min_support: float,
        sep: str = "\t",
        forced_quant_mult: Optional[int] = None,
        allocator: str = "rmm_device",
        pinned: bool = False,
        managed_prefetch: bool = False,
        debug: bool = False,
    ) -> None:
        """Configure I/O, thresholds, allocator pools, and compile kernels."""
        super().__init__(iFile=iFile, debug=debug)
        self.debug = bool(debug)
        self._sep = sep
        self.min_support_float = float(min_support)
        self.forced_quant_mult = forced_quant_mult
        self.scale_factor = 1
        self._minSup_scaled: int = 0
        self._gpu_memory_usage = 0
        self.patterns: Dict[Tuple[str, ...], float] = {}
        self._managed_prefetch = bool(managed_prefetch)

        self._dev_pool, self._pin_pool = _init_allocator(allocator, use_pinned=pinned)

        # Peaks (optional; still computed for print_results)
        self._peak_driver_used = 0
        self._peak_pool_used = 0
        self._peak_pool_total = 0
        self._rmm_peak_bytes = None
        try:
            import rmm.statistics as rstats  # type: ignore
            rstats.enable_statistics()
            self._rmm_stats = rstats
        except Exception:
            self._rmm_stats = None

        # Manual “theory” accounting storage
        self._theory_static_map: Dict[str, int] = {}
        self._theory_trace: List[Dict[str, Any]] = []

        self._kern_num, self._kern_write, self._kern_build, self._kern_mine, self._kern_write_dense = _compile_kernels()

    # -------------------
    # Manual memory helpers (deterministic sizes)
    # -------------------
    @staticmethod
    def _bytes(arr) -> int:
        """Return array bytes or 0 if missing."""
        try:
            return int(arr.nbytes)
        except Exception:
            return 0

    def _static_bytes_post_build(self) -> Dict[str, int]:
        """Sizes of long-lived structures after build() (exact .nbytes, no pooling)."""
        return {
            "buckets": self._bytes(getattr(self, "buckets", None)),
            "dense": self._bytes(getattr(self, "dense", None)),
            "ht_dev": self._bytes(getattr(self, "_ht_dev", None)),
            "arr_counters": self._bytes(getattr(self, "_arr_counters", None)),
            "item_col": self._bytes(getattr(self, "item_col", None)),
            "line_col": self._bytes(getattr(self, "line_col", None)),
            "prob_col": self._bytes(getattr(self, "prob_col", None)),
        }

    @staticmethod
    def _mb(x: int) -> str:
        """Format bytes as MB string with 3 decimals."""
        return f"{x/1024**2:,.3f}"

    def _print_static_breakdown(self, static_map: Dict[str, int]) -> None:
        """Print one-time static memory breakdown after build, if debug is enabled."""
        self._theory_static_map = dict(static_map)  # persist for getters
        if not self.debug:
            return
        parts = [f"{k}={self._mb(v)} MB" for k, v in sorted(static_map.items())]
        print("[theory/after_build] " + "  ".join(parts) + f"  | static_total={self._mb(sum(static_map.values()))} MB")

    # ---- central recorder (stores + prints) ----
    def _record_manual_report(
        self,
        tag: str,
        *,
        static_map: Dict[str, int],
        candidates=None,
        gen_tmp_bytes: int = 0,
        supports=None,
        mask=None,
        next_candidates=None,
        frequent=None,
    ) -> None:
        """Record a manual-memory snapshot and (if debug) print it."""
        static_total = sum(static_map.values())
        b = self._bytes
        snapshot = {
            "tag": tag,
            "static": static_total,
            "start": static_total + b(candidates),
            "gen_peak": static_total + b(candidates) + int(gen_tmp_bytes),
            "mine_peak": static_total + b(candidates) + b(next_candidates) + b(supports) + b(mask),
            "steady_post": static_total + b(frequent),
            "components": {
                "candidates": b(candidates),
                "nxt": b(next_candidates),
                "supports": b(supports),
                "mask": b(mask),
                "frequent": b(frequent),
            },
        }
        self._theory_trace.append(snapshot)

        if self.debug:
            def mbv(k): return self._mb(snapshot[k])
            comp = snapshot["components"]
            print(
                f"[theory/{tag}] static={mbv('static')} MB | start={mbv('start')} MB | "
                f"gen_peak={mbv('gen_peak')} MB | mine_peak={mbv('mine_peak')} MB | "
                f"steady_post={mbv('steady_post')} MB"
            )
            print(
                "           components:"
                f" candidates={self._mb(comp['candidates'])} MB"
                f" nxt={self._mb(comp['nxt'])} MB"
                f" supports={self._mb(comp['supports'])} MB"
                f" mask={self._mb(comp['mask'])} MB"
                f" frequent={self._mb(comp['frequent'])} MB"
            )

    # -------- Public getters --------
    def get_theory_memory_bytes(self, kind: str = "peak") -> int:
        """
        Return manual “theory” memory in bytes:
          - 'static': total of long-lived slabs after build
          - 'final' : last steady_post value
          - 'peak'  : max across start/gen_peak/mine_peak/steady_post
        """
        kind = kind.lower()
        if kind not in {"static", "final", "peak"}:
            raise ValueError("kind must be one of {'static','final','peak'}")

        # Compute static if not captured yet
        static_total = sum(self._theory_static_map.values()) or sum(self._static_bytes_post_build().values())

        if not self._theory_trace:
            return static_total if kind != "peak" else static_total

        if kind == "static":
            return static_total
        if kind == "final":
            return int(self._theory_trace[-1]["steady_post"])

        # peak
        peak = 0
        for s in self._theory_trace:
            peak = max(peak, int(s["start"]), int(s["gen_peak"]), int(s["mine_peak"]), int(s["steady_post"]))
        return peak

    def get_theory_memory_report(self) -> Dict[str, Any]:
        """Return a dict with static_total, final_total, peak_total, and the full per-iteration trace."""
        static_total = sum(self._theory_static_map.values()) or sum(self._static_bytes_post_build().values())
        final_total = self.get_theory_memory_bytes("final")
        peak_total = self.get_theory_memory_bytes("peak")
        return {
            "static_total_bytes": static_total,
            "final_total_bytes": final_total,
            "peak_total_bytes": peak_total,
            "trace": copy.deepcopy(self._theory_trace),
        }

    # -------------------
    # Optional telemetry peaks (for summary only)
    # -------------------
    def _snap_mem(self) -> None:
        """Update peak counters using driver/pool info; used only for summary at the end."""
        try:
            cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass
        try:
            free_b, total_b = cp.cuda.runtime.memGetInfo()
            used_b = total_b - free_b
            self._peak_driver_used = max(self._peak_driver_used, used_b)
        except Exception:
            pass
        try:
            pool = cp.get_default_memory_pool()
            self._peak_pool_used = max(self._peak_pool_used, pool.used_bytes())
            self._peak_pool_total = max(self._peak_pool_total, pool.total_bytes())
        except Exception:
            pass
        try:
            if self._rmm_stats is not None:
                stats = self._rmm_stats.get_statistics()
                self._rmm_peak_bytes = getattr(stats, "peak_bytes", None)
        except Exception:
            pass

    # -------------------
    # Data loading & prep
    # -------------------
    def _cpu_load_and_scale_data(self) -> pl.DataFrame:
        """Load text/parquet with Polars, explode rows, and quantize probabilities to UInt32."""
        path = str(self._iFile)
        if path.lower().endswith(".parquet"):
            df = pl.read_parquet(path)
            prob_col = "prob"
        else:
            lazy_txns = pl.scan_csv(
                path, has_header=False, separator=":", new_columns=["items_str", "values_str"]
            ).with_row_index("txn_id", offset=1)
            df = (
                lazy_txns.with_columns(
                    pl.col("items_str").str.split(self._sep),
                    pl.col("values_str").str.split(self._sep),
                )
                .filter(pl.col("items_str").list.len() == pl.col("values_str").list.len())
                .explode(["items_str", "values_str"])
                .rename({"items_str": "item", "values_str": "prob"})
            ).collect()
            prob_col = "prob"

        if self.forced_quant_mult is not None:
            self.scale_factor = int(self.forced_quant_mult)
        else:
            max_decimals = (
                df.get_column(prob_col)
                .cast(pl.Utf8)
                .str.split_exact(".", 1)
                .struct.field("field_1")
                .str.len_chars()
                .fill_null(0)
                .max()
            )
            self.scale_factor = 10 ** int(max_decimals or 0)

        self._minSup_scaled = int(self.min_support_float * self.scale_factor)

        df_scaled = df.with_columns(
            (pl.col(prob_col).cast(pl.Float64) * self.scale_factor).cast(pl.UInt32).alias("prob")
        ).select(pl.col("item").cast(pl.Utf8), pl.col("prob"), pl.col("txn_id").cast(pl.UInt32))
        return df_scaled

    def _calculate_support_and_filter_items(self, df: pl.DataFrame):
        """Compute item supports, filter by threshold, and assign dense IDs; return (support_df, final_df)."""
        support_df = df.group_by("item").agg(pl.sum("prob"), pl.len().alias("freq"))
        support_df = support_df.filter(pl.col("prob") >= self._minSup_scaled).sort("freq", descending=True)
        if len(support_df) == 0:
            return None, None
        n_items = len(support_df)
        support_df = support_df.with_columns(
            new_id=pl.arange(FIRST_ID, FIRST_ID + n_items, dtype=pl.UInt32, eager=True)
        )
        self.max_id = support_df["new_id"].max()
        if self.max_id is None:
            return None, None
        self.rename_map = np.empty(self.max_id + 1, dtype=object)
        self.rename_map[support_df["new_id"].to_numpy()] = support_df["item"].to_numpy()
        final_df = df.join(support_df.select("item", "new_id"), on="item").drop("item").rename({"new_id": "item"})
        self.support_df = support_df
        return support_df, final_df

    @staticmethod
    def _to_device_struct(arr: np.ndarray):
        """Cast a NumPy structured array to a CuPy view preserving dtype layout."""
        return cp.asarray(arr.view(np.uint8)).view(arr.dtype)

    def _allocate_gpu_hash_tables(self, support_df: pl.DataFrame):
        """Allocate per-item open-addressing tables and dense arrays as contiguous slabs; return host descriptors."""
        freq_cp = cp.asarray(support_df["freq"].to_numpy(), dtype=cp.uint32)
        needed = (freq_cp * INV_LOAD_FACTOR).astype(cp.float32)
        sizes_cp = cp.power(2, cp.ceil(cp.log2(needed))).astype(cp.uint32)
        slots_total = int(sizes_cp.sum())

        # Buckets slab
        self.buckets = cp.zeros(slots_total, dtype=KVPair)
        base_ptr = self.buckets.data.ptr
        offset = cp.cumsum(cp.concatenate([cp.array([0], dtype=cp.uint64), sizes_cp.astype(cp.uint64)])) * KVPair.itemsize

        # Dense slab
        dense_total = int(freq_cp.sum())
        self.dense = cp.empty(dense_total, dtype=KVPair)
        dense_base = self.dense.data.ptr
        dense_offs = cp.cumsum(cp.concatenate([cp.array([0], dtype=cp.uint64), freq_cp.astype(cp.uint64)])) * KVPair.itemsize

        # Host descriptors
        ht_host = np.zeros(self.max_id + 1, dtype=HashTable)
        sizes_h = sizes_cp.get()
        offset_h = offset.get()
        freq_h = freq_cp.get()
        dense_off = dense_offs.get()

        for idx, id_ in enumerate(support_df["new_id"].to_numpy()):
            ht_host[id_]["table_ptr"] = base_ptr + offset_h[idx]
            ht_host[id_]["table_size"] = sizes_h[idx]
            ht_host[id_]["mask"] = sizes_h[idx] - 1
            ht_host[id_]["id"] = id_
            ht_host[id_]["array_ptr"] = dense_base + dense_off[idx]
            ht_host[id_]["array_size"] = freq_h[idx]

        self._max_threads = min(int(sizes_cp.min()), THREADS) if len(sizes_cp) > 0 else THREADS
        self._snap_mem()
        return ht_host

    def _build_gpu_hash_tables(self, ht_host: np.ndarray, final_df: pl.DataFrame) -> None:
        """Transfer columns to GPU, build hash tables, and populate dense arrays."""
        self.item_col = cp.asarray(final_df["item"].to_numpy(), dtype=cp.uint32)
        self.line_col = cp.asarray(final_df["txn_id"].to_numpy(), dtype=cp.uint32)
        self.prob_col = cp.asarray(final_df["prob"].to_numpy(), dtype=cp.uint64)
        self._ht_dev = self._to_device_struct(ht_host)

        total_pairs = len(self.item_col)
        grid = (total_pairs + self._max_threads - 1) // self._max_threads

        self._kern_build(
            (grid,), (self._max_threads,), (total_pairs, self.item_col, self.line_col, self.prob_col, self._ht_dev)
        )
        self._arr_counters = cp.zeros(self.max_id + 1, dtype=cp.uint32)
        self._kern_write_dense(
            (grid,),
            (self._max_threads,),
            (total_pairs, self.item_col, self.line_col, self.prob_col, self._ht_dev, self._arr_counters),
        )
        cp.cuda.runtime.deviceSynchronize()
        self._snap_mem()

        if self._managed_prefetch:
            try:
                dev = cp.cuda.runtime.getDevice()
                cp.cuda.runtime.memPrefetchAsync(self.buckets.data.ptr, self.buckets.nbytes, dev, 0)
                cp.cuda.runtime.memPrefetchAsync(self.dense.data.ptr, self.dense.nbytes, dev, 0)
            except Exception:
                pass

    # -------------------
    # Candidate generation (returns temps size)
    # -------------------
    def _generate_next_candidates(self, cands: cp.ndarray, k: int):
        """Generate (k+1)-item candidates and return (array, temp_bytes_during_generation)."""
        n = len(cands)
        if n == 0:
            return cp.array([], dtype=cp.uint32), 0
        grid = (n + self._max_threads - 1) // self._max_threads
        num = cp.zeros(n + 1, dtype=cp.uint32)  # temp
        self._kern_num((grid,), (self._max_threads,), (cands, n, k, num))
        idx = cp.cumsum(num, dtype=cp.uint32)  # temp
        out_len = int(idx[-1].get())
        if out_len == 0:
            gen_tmp_bytes = int(num.nbytes + idx.nbytes)
            return cp.array([], dtype=cp.uint32), gen_tmp_bytes
        nxt = cp.zeros((out_len, k + 1), dtype=cp.uint32)  # temp
        self._kern_write((grid,), (self._max_threads,), (cands, n, k, idx, nxt))
        gen_tmp_bytes = int(num.nbytes + idx.nbytes + nxt.nbytes)
        return nxt, gen_tmp_bytes

    # -------------------
    # Main pipeline
    # -------------------
    def _mine(self) -> None:
        """Run load→filter→build→iterate→collect with manual memory accounting."""
        path = Path(self._iFile)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        _t0 = time.time()
        exploded_df = self._cpu_load_and_scale_data()
        support_df, final_df = self._calculate_support_and_filter_items(exploded_df)
        if support_df is None:
            self.patterns = {}
            return

        ht_host = self._allocate_gpu_hash_tables(support_df)
        self._build_gpu_hash_tables(ht_host, final_df)
        del exploded_df, support_df, final_df
        gc.collect()

        # Static breakdown after build
        static_map = self._static_bytes_post_build()
        self._print_static_breakdown(static_map)

        # Iterate (manual memory accounting per phase)
        k = 1
        candidates = cp.asarray(self.support_df["new_id"].to_numpy()).reshape(-1, 1)
        gpu_results = []
        while len(candidates) > 0:
            print(f"--- k={k} candidates: {len(candidates)} ---")

            # START state (only static + current candidates)
            self._record_manual_report(f"k={k}/start", static_map=static_map, candidates=candidates)

            # Candidate generation
            next_candidates, gen_tmp_bytes = self._generate_next_candidates(candidates, k)
            self._record_manual_report(
                f"k={k}/gen",
                static_map=static_map,
                candidates=candidates,
                gen_tmp_bytes=gen_tmp_bytes,
                next_candidates=next_candidates,
            )

            if len(next_candidates) == 0:
                break

            # Mining
            k += 1
            supports = cp.zeros(len(next_candidates), dtype=cp.uint64)
            self._kern_mine(
                (len(next_candidates),),
                (self._max_threads,),
                (self._ht_dev, next_candidates, len(next_candidates), k, supports),
            )
            cp.cuda.runtime.deviceSynchronize()

            mask = supports >= self._minSup_scaled
            frequent = next_candidates[mask]

            # Report mining peak (cands + next + supports + mask coexist)
            self._record_manual_report(
                f"k={k}/mine",
                static_map=static_map,
                candidates=candidates,
                next_candidates=next_candidates,
                supports=supports,
                mask=mask,
                frequent=frequent,
            )

            # Collect results
            if len(frequent) > 0:
                gpu_results.append((frequent.get(), supports[mask].get()))

            # Steady post (only static + frequent)
            self._record_manual_report(f"k={k}/post", static_map=static_map, frequent=frequent)

            # Free temps and advance
            del supports, mask, next_candidates
            gc.collect()
            candidates = frequent

        self._process_results(gpu_results)

        # Summary peaks for print_results()
        self._snap_mem()
        if self._rmm_stats is not None:
            try:
                stats = self._rmm_stats.get_statistics()
                self._rmm_peak_bytes = getattr(stats, "peak_bytes", None)
            except Exception:
                pass

        self._gpu_memory_usage = self._peak_driver_used

        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.set_pinned_memory_allocator(None)
            cp.cuda.set_allocator(None)
        except Exception:
            pass

    # -------------------
    # Results & I/O
    # -------------------
    def _process_results(self, gpu_results) -> None:
        """Assemble singleton and higher-order itemset supports into a Python dict."""
        patterns = {
            (item,): float(prob) / float(self.scale_factor)
            for item, prob in zip(self.support_df["item"].to_numpy(), self.support_df["prob"].to_numpy())
        }
        for ids_cpu, vals in gpu_results:
            names = self.rename_map[ids_cpu]
            for name_tuple, sup in zip(names, vals):
                patterns[tuple(name_tuple)] = float(sup) / float(self.scale_factor)
        self.patterns = patterns

    def save(self, oFile: Union[str, Path]) -> None:
        """Write mined itemsets and supports to a TSV file."""
        with open(oFile, "w") as f:
            for itemset, support in self.patterns.items():
                f.write(f"{','.join(itemset)}\t{support}\n")

    def print_results(self) -> None:
        """Print concise runtime and memory telemetry plus pattern count."""
        theory_static_mb = self._mb(self.get_theory_memory_bytes("static"))
        theory_peak_mb = self._mb(self.get_theory_memory_bytes("peak"))

        print(f"\n--- {self.__class__.__name__} Results ---")
        print(f"Execution Time: {self.get_execution_time():.4f} seconds")
        print(f"Peak CPU Memory Usage: {self.get_memory_usage():.2f} MB")
        print(f"Peak GPU (driver) Used: {self._peak_driver_used / (1024**2):.2f} MB")
        print(f"Peak Pool Used:        {self._peak_pool_used / (1024**2):.2f} MB")
        print(f"Peak Pool Total:       {self._peak_pool_total / (1024**2):.2f} MB")
        if self._rmm_peak_bytes is not None:
            print(f"RMM Statistics Peak:   {self._rmm_peak_bytes / (1024**2):.2f} MB")
        print(f"Theoretical Static:    {theory_static_mb} MB")
        print(f"Theoretical Peak:      {theory_peak_mb} MB")
        print(f"Patterns Found: {self.get_pattern_count()}")
        print("--------------------" + "-" * len(self.__class__.__name__))


# -----------------------
# CLI
# -----------------------
def _cli() -> None:
    """Parse arguments, run the miner, and output results."""
    p = argparse.ArgumentParser(description="naiveFFIMiner (manual-memory) using Polars and CUDA")
    p.add_argument("iFile", type=str, help="Path to input (.txt/.csv/.tsv or .parquet)")
    p.add_argument("min_support", type=float, help="Minimum support threshold (e.g., 0.1 for 10%)")
    p.add_argument("-o", "--oFile", type=str, default="patterns.txt", help="Path to output file")
    p.add_argument("--sep", type=str, default="\t", help="Separator for items in text inputs")
    p.add_argument(
        "--allocator",
        type=str,
        default="rmm_device",
        choices=["rmm_device", "rmm_managed", "cupy_device", "cupy_managed"],
        help="GPU allocator/pool mode",
    )
    p.add_argument("--pinned", action="store_true", help="Enable pinned host memory pool")
    p.add_argument("--managed-prefetch", action="store_true", help="Prefetch managed slabs to device")
    p.add_argument("--forced-quant-mult", type=int, default=None, help="Force integer scaling factor for probabilities")
    p.add_argument("--debug", action="store_true", help="Print manual memory accounting per iteration")
    args = p.parse_args()

    miner = naiveFFIMiner(
        iFile=args.iFile,
        min_support=args.min_support,
        sep=args.sep,
        forced_quant_mult=args.forced_quant_mult,
        allocator=args.allocator,
        pinned=args.pinned,
        managed_prefetch=args.managed_prefetch,
        debug=args.debug,
    )
    miner.mine()
    miner.save(args.oFile)
    miner.print_results()


if __name__ == "__main__":  # pragma: no cover
    _cli()
