"""
CUDA Bitset-based Frequent Pattern Miner
====================================================

Highlights in this refactor
---------------------------
- Replaced hash sets with bitsets for maximum parallelism.
- Support is calculated via bitwise AND operations across bitsets.
- Uses CUDA's __popc() intrinsic for highly efficient frequency counting.
- Retained the parallel sum reduction for aggregating block-level supports.
- Kept the single, switchable allocator, I/O toggles, and memory telemetry.

Note: This approach is most effective when the maximum transaction ID is
manageable, as it determines the size of the bitsets.
"""
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Union, Dict, Tuple, Optional
import re
import time
import numpy as np
import cupy as cp


from ..base import BaseAlgorithm  # type: ignore

# -----------------------
# Global knobs & helpers
# -----------------------
THREADS = 1024
FIRST_ID = 1

# -----------------------
# CUDA kernel source
# -----------------------
KERNEL_SRC = r"""
// ------------------------------------------------------------
// Common helpers & data structures
// ------------------------------------------------------------
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#define FIRST_ID 1u

// Candidate generation (unchanged)
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

// NEW: Kernel to build bitsets from item-transaction pairs
extern "C" __global__ void build_bitsets(
    uint64_t total_pairs,
    const uint32_t* __restrict__ items, // new_id
    const uint32_t* __restrict__ lines, // txn_id
    uint32_t bitset_len_in_uints,
    uint32_t num_frequent_items,
    uint32_t* __restrict__ bitsets
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= total_pairs) return;

    uint32_t item_id = items[tid]; // is 1-based new_id
    uint32_t txn_id = lines[tid];  // is 1-based

    // Make IDs 0-based for array/bit indexing
    uint32_t bit_pos = (txn_id - 1) % 32;
    uint32_t word_pos = (txn_id - 1) / 32;

    // new_id is 1-based, our bitset array is 0-indexed
    uint32_t bitset_row_idx = item_id - FIRST_ID;
    uint32_t word_global_idx = bitset_row_idx * bitset_len_in_uints + word_pos;

    atomicOr(&bitsets[word_global_idx], (1u << bit_pos));
}


// NEW: Kernel to mine candidates using bitsets
extern "C" __global__ void mine_kernel(
    const uint32_t* __restrict__ bitsets,
    uint32_t bitset_len_in_uints,
    const uint32_t*   __restrict__ cands,
    uint32_t nCands, uint32_t candSize,
    uint64_t* __restrict__ supports
){
    uint32_t bid = blockIdx.x;
    if (bid >= nCands) return;

    extern __shared__ uint32_t s_support[];
    uint32_t tid = threadIdx.x;
    s_support[tid] = 0;

    // Each thread processes a chunk of the bitset's uint32 words
    for (uint32_t i = tid; i < bitset_len_in_uints; i += blockDim.x) {
        uint32_t result_word = 0xFFFFFFFF; // Start with all 1s for AND operation

        // Intersect (bitwise AND) the bitsets for all items in the candidate
        #pragma unroll 1
        for (uint32_t j = 0; j < candSize; ++j) {
            uint32_t item_id = cands[bid * candSize + j];
            uint32_t bitset_row_idx = item_id - FIRST_ID; // new_id is 1-based
            result_word &= bitsets[bitset_row_idx * bitset_len_in_uints + i];
        }

        // If there are any common transactions in this word, count them
        if (result_word != 0) {
            s_support[tid] += __popc(result_word);
        }
    }
    __syncthreads();

    // Parallel sum reduction to get total support for the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_support[tid] += s_support[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result for this candidate
    if (tid == 0 && s_support[0] > 0) {
        atomicAdd(&supports[bid], (unsigned long long)s_support[0]);
    }
}
"""

# -----------------------
# Allocator & IO toggles
# -----------------------

def _set_kvikio_mode(mode: str) -> None:
    """KVIKIO_COMPAT_MODE controls cuDF/KvikIO I/O path.
    mode in {"auto","on","off"}
      - auto: try cuFile (GDS), fallback to POSIX
      - on:   force POSIX compatibility (GDS OFF)
      - off:  force cuFile path (GDS ON)
    """
    m = mode.lower()
    if m not in {"auto", "on", "off"}:
        raise ValueError("gds must be one of: auto, on, off")
    os.environ["KVIKIO_COMPAT_MODE"] = m.upper()


def _init_allocator(mode: str, use_pinned: bool=False) -> Tuple[Optional[object], Optional[object]]:
    """Initialize a *single* allocator for this process.
    Returns (device_pool, pinned_pool). Device pool may be an RMM resource or CuPy MemoryPool.
    mode:
      - "rmm_device"   -> RMM pool over cudaMalloc
      - "rmm_managed"  -> RMM pool over cudaMallocManaged (UVM)
      - "cupy_device"  -> CuPy MemoryPool
      - "cupy_managed" -> CuPy MemoryPool (managed)
    """
    
    dev_pool = pin_pool = None
    mode = mode.lower()
    if mode.startswith("rmm"):
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator
        if mode == "rmm_device":
            mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource())
        elif mode == "rmm_managed":
            mr = rmm.mr.PoolMemoryResource(rmm.mr.ManagedMemoryResource())
        else:
            raise ValueError("Unknown allocator mode: " + mode)
        rmm.mr.set_current_device_resource(mr)
        cp.cuda.set_allocator(rmm_cupy_allocator)
        dev_pool = mr
    elif mode == "cupy_device":
        dev_pool = cp.cuda.MemoryPool()
        cp.cuda.set_allocator(dev_pool.malloc)
    elif mode == "cupy_managed":
        dev_pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
        cp.cuda.set_allocator(dev_pool.malloc)
    else:
        raise ValueError("Unknown allocator mode: " + mode)

    if use_pinned:
        pin_pool = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(pin_pool.malloc)
    return dev_pool, pin_pool


# -----------------------
# GPU compile (lazy)
# -----------------------

def _compile_kernels():
    NVCC_OPTIONS = ("-std=c++14","-O3","-Xptxas=-v","-Xptxas=--warn-on-spills","-Xptxas=-dlcm=ca","-lineinfo")
    mod = cp.RawModule(code=KERNEL_SRC, backend="nvcc", options=NVCC_OPTIONS)
    return (
        mod.get_function("number_of_new_candidates_to_generate"),
        mod.get_function("write_the_new_candidates"),
        mod.get_function("build_bitsets"),
        mod.get_function("mine_kernel"),
    )



# -----------------------
# Miner implementation
# -----------------------
class cuBPMiner(BaseAlgorithm):
    """
    cuDF/CUDA-based bitset frequent pattern miner.

    Constructor knobs:
      - allocator: one of {rmm_device, rmm_managed, cupy_device, cupy_managed}
      - pinned:    bool to enable a pinned host pool
      - gds:       {auto,on,off} to control I/O path
      - managed_prefetch: bool, prefetch large managed arrays to device
    """

    def __init__(
        self,
        iFile: Union[str, Path],
        min_support: int,
        sep: str = "\t",
        allocator: str = "rmm_device",
        pinned: bool = False,
        gds: str = "auto",
        managed_prefetch: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(iFile=iFile, debug=debug)
        self._sep = sep
        self._minSup = int(min_support)
        self._gpu_memory_usage = 0
        self.patterns: Dict[Tuple[str, ...], int] = {}
        self._managed_prefetch = bool(managed_prefetch)
        self._max_threads = THREADS

        # I/O path first (affects subsequent reads)
        _set_kvikio_mode(gds)

        # Unified allocator for both cuDF & CuPy
        self._dev_pool, self._pin_pool = _init_allocator(allocator, use_pinned=pinned)

        # Telemetry
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

        # Kernels
        self._kern_num, self._kern_write, self._kern_build, self._kern_mine = _compile_kernels()

    # -------------------
    # Memory snapshots
    # -------------------
    def _snap_mem(self) -> None:
        
        cp.cuda.runtime.deviceSynchronize()
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

    # -------------------
    # Data loading
    # -------------------
    def _load_text(self, path: Union[str, Path]):
        import cudf
        gdf = cudf.read_csv(str(path), sep=":", header=None,
                            names=["items"], dtype=["str"])
        sep_esc = re.escape(self._sep)
        pattern = rf"[{sep_esc}\r\n ]+$"
        gdf["items"] = gdf["items"].str.replace(pattern, "", regex=True)
        gdf["items"] = gdf["items"].str.split(self._sep)
        gdf = gdf.reset_index(drop=True)
        gdf["txn_id"] = (gdf.index + 1).astype("uint32")
        return gdf

    def _explode_transactions(self, gdf):
        items_ex = gdf[["txn_id", "items"]].explode("items")
        return items_ex.rename(columns={"items": "item"})

    def _load_parquet(self, path: Union[str, Path]):
        import cudf
        gdf = cudf.read_parquet(str(path))
        req = ["item", "txn_id"]
        if not all(c in gdf.columns for c in req):
            raise ValueError(f"Parquet must contain columns: {req}")
        gdf["item"] = gdf["item"].astype("str")
        gdf["txn_id"] = gdf["txn_id"].astype("uint32")
        return gdf

    # -------------------
    # Support & filtering
    # -------------------
    def _calculate_support_and_filter_items(self, gdf):
        # CRITICAL: Get max transaction ID *before* filtering to size bitsets correctly
        self.max_tid = int(gdf["txn_id"].max())
        
        support_df = gdf.groupby("item").agg({"txn_id": "count"}).rename(columns={"txn_id": "freq"}).reset_index()
        support_df = support_df[support_df["freq"] >= self._minSup].sort_values("freq", ascending=False).reset_index(drop=True)

        if len(support_df) == 0:
            return None, None
            
        self.num_frequent_items = len(support_df)
        support_df["new_id"] = cp.arange(FIRST_ID, FIRST_ID + self.num_frequent_items, dtype=cp.uint32)
        
        self.rename_map = np.empty(self.num_frequent_items + 1, dtype=object)
        self.rename_map[support_df["new_id"].to_numpy()] = support_df["item"].to_numpy()
        
        gdf = gdf[gdf["item"].isin(support_df["item"])]
        gdf = gdf.merge(support_df[["item", "new_id"]], on="item")
        gdf = gdf.drop(columns=["item"]).rename(columns={"new_id": "item"})
        self.support_df = support_df
        return support_df, gdf

    # -------------------
    # GPU structures
    # -------------------
    def _allocate_gpu_bitsets(self):
        # Calculate the length of each bitset in terms of 32-bit integers
        self.bitset_len_in_uints = (self.max_tid + 31) // 32
        
        # Allocate a 2D array: (num_frequent_items x bitset_length)
        self.bitsets = cp.zeros((self.num_frequent_items, self.bitset_len_in_uints), dtype=cp.uint32)
        self._snap_mem()


    def _build_gpu_bitsets(self, gdf):
        item_col = gdf["item"].astype("uint32").values
        line_col = gdf["txn_id"].astype("uint32").values

        total_pairs = len(item_col)
        grid_size = (total_pairs + self._max_threads - 1) // self._max_threads

        # Launch kernel to populate the bitsets
        self._kern_build(
            (grid_size,),
            (self._max_threads,),
            (total_pairs, item_col, line_col, self.bitset_len_in_uints, self.num_frequent_items, self.bitsets)
        )

        cp.cuda.runtime.deviceSynchronize()
        self._snap_mem()

        if self._managed_prefetch:
            try:
                dev = cp.cuda.runtime.getDevice()
                cp.cuda.runtime.memPrefetchAsync(self.bitsets.data.ptr, self.bitsets.nbytes, dev, 0)
            except Exception:
                pass

    # -------------------
    # Main mining loop
    # -------------------
    def _generate_next_candidates(self, cands, k: int):
        
        n = len(cands)
        if n == 0:
            return cp.array([], dtype=cp.uint32)
        grid_size = (n + self._max_threads - 1) // self._max_threads
        num = cp.zeros(n + 1, dtype=cp.uint32)
        self._kern_num((grid_size,), (self._max_threads,), (cands, n, k, num))
        idx = cp.cumsum(num, dtype=cp.uint32)
        out_len = int(idx[-1].get())
        if out_len == 0:
            return cp.array([], dtype=cp.uint32)
        nxt = cp.zeros((out_len, k + 1), dtype=cp.uint32)
        self._kern_write((grid_size,), (self._max_threads,), (cands, n, k, idx, nxt))
        return nxt

    def _mine(self):
        path = Path(self._iFile)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        t0 = time.time()
        if path.suffix.lower() == ".parquet":
            exploded_gdf = self._load_parquet(path)
        else:
            raw_gdf = self._load_text(path)
            exploded_gdf = self._explode_transactions(raw_gdf)
        self._snap_mem()

        support_df, final_gdf = self._calculate_support_and_filter_items(exploded_gdf)
        if support_df is None:
            self.patterns = {}
            return

        self._allocate_gpu_bitsets()
        self._build_gpu_bitsets(final_gdf)
        del exploded_gdf, support_df, final_gdf

        k = 1
        
        candidates = self.support_df["new_id"].to_cupy().reshape(-1, 1)
        gpu_results = []
        while len(candidates) > 0:
            next_candidates = self._generate_next_candidates(candidates, k)
            if len(next_candidates) == 0:
                break
            k += 1
            supports = cp.zeros(len(next_candidates), dtype=cp.uint64)
            self._kern_mine(
                (len(next_candidates),), 
                (self._max_threads,), 
                (self.bitsets, self.bitset_len_in_uints, next_candidates, len(next_candidates), k, supports), 
                shared_mem=self._max_threads * 4 # 4 bytes per thread for shared memory reduction
            )
            cp.cuda.runtime.deviceSynchronize()
            self._snap_mem()
            mask = supports >= self._minSup
            frequent_cands = next_candidates[mask]
            if len(frequent_cands) > 0:
                gpu_results.append((frequent_cands.get(), supports[mask].get()))
            candidates = frequent_cands

        self._process_results(gpu_results)

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
    # Results / I/O
    # -------------------
    def _process_results(self, gpu_results):
        patterns = {
            (item,): int(freq)
            for item, freq in zip(
                self.support_df["item"].to_numpy(),
                self.support_df["freq"].to_numpy(),
            )
        }
        for ids_cpu, vals in gpu_results:
            names = self.rename_map[ids_cpu]
            for name_tuple, sup in zip(names, vals):
                patterns[tuple(name_tuple)] = int(sup)
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


# -----------------------
# CLI entry
# -----------------------

def _cli():
    p = argparse.ArgumentParser(description="cuBPMiner for bitset-based frequent pattern mining")
    p.add_argument("iFile", type=str, help="Path to input (.txt/.csv/.tsv or .parquet)")
    p.add_argument("min_support", type=int, help="Minimum support threshold (frequency)")
    p.add_argument("-o", "--oFile", type=str, default="patterns.txt")
    p.add_argument("--sep", type=str, default="\t", help="Separator for items in text input files")
    p.add_argument(
        "--allocator", type=str, default="rmm_device",
        choices=["rmm_device", "rmm_managed", "cupy_device", "cupy_managed"],
        help="GPU allocator/pool mode"
    )
    p.add_argument("--pinned", action="store_true", help="Enable pinned host memory pool")
    p.add_argument("--gds", type=str, default="auto", choices=["auto", "on", "off"], help="GDS I/O path toggle")
    p.add_argument("--managed-prefetch", action="store_true", help="Prefetch managed slabs to device")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    miner = cuBPMiner(
        iFile=args.iFile,
        min_support=args.min_support,
        sep=args.sep,
        allocator=args.allocator,
        pinned=args.pinned,
        gds=args.gds,
        managed_prefetch=args.managed_prefetch,
        debug=args.debug,
    )
    miner.mine()
    miner.save(args.oFile)
    miner.print_results()

if __name__ == "__main__":
    _cli()