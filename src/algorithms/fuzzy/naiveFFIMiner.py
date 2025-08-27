#!/usr/bin/env python3
"""
naiveFFIMiner - Advanced CUDA-based Fuzzy Frequent Itemset Miner
===================================================================

This module implements an advanced version of the naiveFFIMiner algorithm.

Key Features:
-------------
1.  **Automatic Scaling**: Accepts floating-point fuzzy values and `min_support`.
    It automatically scans the data to determine the optimal integer scaling
    factor, preserving precision.

2.  **High-Performance CPU Pre-processing**: Leverages the Polars library to use
    all available CPU cores for ultra-fast loading and transformation of text (CSV/TSV)
    and Parquet files before handing the data to the GPU.

3.  **Efficient GPU Mining**: The core mining logic remains on the GPU, using the
    same proven CUDA kernels for maximum throughput.

Dependencies:
-------------
This script requires Polars, PyArrow, and the GPU libraries.
- pip install polars pyarrow
- pip install cudf-cu11 cupy-cuda11x  # Or version matching your CUDA toolkit
"""

import os
import argparse
from pathlib import Path
from typing import Union, Dict, Tuple
import re

# Import everything from the base module.
from ..base import *

os.environ["POLARS_MAX_THREADS"] = str(os.cpu_count())  # set BEFORE importing polars


# Check for GPU and CPU library availability
try:
    import cupy as cp
    import cudf
    import polars as pl
    GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    print("Warning: Missing one or more required libraries (cupy, cudf, polars).")
    GPU_AVAILABLE = False

# --- CUDA & Host Data Structures (unchanged) ---
THREADS = 1024
INV_LOAD_FACTOR = 3/2
KVPair = np.dtype([("line", np.uint32), ("probability", np.uint64)], align=True)
HashTable = np.dtype([("table_ptr", np.uintp), ("table_size", np.uint32), ("mask", np.uint32), ("id", np.uint32)], align=True)
assert KVPair.itemsize == 16, "Host KVPair layout must be 16 Bytes"
assert HashTable.itemsize == 24, "Host HashTable layout must be 24 Bytes"

def _to_device_struct(arr: np.ndarray) -> cp.ndarray:
    return cp.asarray(arr.view(np.uint8)).view(arr.dtype)

SENTINEL = 0
FIRST_ID = 1

# --- CUDA Kernel Source Code (unchanged) ---
KERNEL_SRC = r"""
// ... (CUDA kernel code remains the same as the previous versions) ...
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

# --- Compile CUDA Kernels ---
if GPU_AVAILABLE:
    NVCC_OPTIONS = [
    "-std=c++14",
    "-O3",
    "-Xptxas=-v",                  # show regs/spills to guide tuning
    "-Xptxas=--warn-on-spills",
    "-Xptxas=-dlcm=ca",            # prefer caching global loads in L1+L2
    "-lineinfo",                   # preserve line info for Nsight Compute
        ]
    RAWMOD = cp.RawModule(code=KERNEL_SRC, backend="nvcc", options=tuple(NVCC_OPTIONS))
    # ... (Kernel function getters remain the same) ...
    KERN_NUM = RAWMOD.get_function("number_of_new_candidates_to_generate")
    KERN_WRITE = RAWMOD.get_function("write_the_new_candidates")
    KERN_BUILD = RAWMOD.get_function("build_tables")
    KERN_MINE = RAWMOD.get_function("mine_kernel")
else:
    RAWMOD = KERN_NUM = KERN_WRITE = KERN_BUILD = KERN_MINE = None


class naiveFFIMiner(BaseAlgorithm):
    """Advanced implementation of cuFFIMiner with automatic scaling and high-performance
    CPU-based pre-processing using the Polars library.

    This class discovers fuzzy frequent itemsets from transactional data. It is optimized
    for performance by using a combination of CPU and GPU processing. The CPU is used for
    efficiently loading and pre-processing the data with Polars, while the GPU accelerates
    the core mining process using CUDA kernels.

    A key feature is its ability to automatically determine the correct scaling factor
    for floating-point fuzzy values, which simplifies configuration and ensures
    precision is maintained during the mining process.

    :ivar patterns: A dictionary storing the discovered frequent patterns. The keys are
                    tuples of item names, and the values are their corresponding support
                    as floating-point numbers.
    :vartype patterns: Dict[Tuple[str, ...], float]
    """

    def __init__(self, iFile: Union[str, Path], min_support: float, sep: str = "\t", quant_mult: int = None, debug: bool = False):
        """Initializes the advanced cuFFIMiner algorithm.

        :param iFile: Path to the input file. Can be a text file (CSV, TSV) or a Parquet file.
        :type iFile: Union[str, Path]
        :param min_support: The minimum support threshold as a float (e.g., 0.05 for 5%).
        :type min_support: float
        :param sep: The separator character used in text input files. Defaults to tab.
        :type sep: str, optional
        :param debug: If True, enables detailed memory and CPU profiling messages.
        :type debug: bool, optional
        :raises RuntimeError: If required GPU libraries (cuDF, CuPy) or Polars are not available.
        """
        super().__init__(iFile=iFile, debug=debug)

        if debug:
            pl.Config.set_verbose(True)


        if not GPU_AVAILABLE:
            raise RuntimeError("cuFFIMiner_Adv requires GPU libraries (cuDF, CuPy) and Polars.")

        self.min_support_float = min_support
        # Optional externally provided quantization multiplier (forced)
        self.forced_quant_mult = quant_mult if (quant_mult is not None and quant_mult > 0) else None
        self._sep = sep
        self._gpu_memory_usage = 0
        self.patterns: Dict[Tuple[str, ...], float] = {}

    def _cpu_load_and_scale_data(self) -> cudf.DataFrame:
        """
        Loads, transforms, and scales data using Polars' LAZY engine for maximum parallelism
        and performs a high-performance data handoff to the GPU via Apache Arrow.
        """
        is_parquet = str(self._iFile).lower().endswith(".parquet")

        if is_parquet:
            df = pl.read_parquet(self._iFile)
            value_col_name = "prob"
        else:
            # --- OPTIMIZED LAZY TEXT FILE PROCESSING ---
            # 1. Use scan_csv for a lazy read. This doesn't load data, it just creates a plan.
            #    Read with the primary delimiter FIRST. This is key for parallelization.
            lazy_df_base = pl.scan_csv(
                self._iFile,
                has_header=False,
                separator=":",
                new_columns=["items_str", "values_str"]
            ).with_columns(
                # 2. Build the parsing steps into the lazy plan
                pl.col("items_str").str.split(self._sep),
                pl.col("values_str").str.split(self._sep)
            )

            # 3. Add the filter to the lazy plan. No data is touched yet.
            lazy_df_filtered = lazy_df_base.filter(
                pl.col("items_str").list.len() == pl.col("values_str").list.len()
            )

            # 4. Add the explode and rename operations to the lazy plan.
            lazy_df_exploded = lazy_df_filtered.explode(["items_str", "values_str"]).rename(
                {"items_str": "item", "values_str": "prob"}
            )
            
            # 5. TRIGGER EXECUTION: .collect() tells Polars to run the optimized plan now.
            #    Polars will now read and process the file in parallel chunks.
            df = lazy_df_exploded.with_row_index("txn_id", offset=1).collect()
            value_col_name = "prob"

        if self.forced_quant_mult is not None:
            self.scale_factor = int(self.forced_quant_mult)
            max_decimals = None
        else:
            max_decimals = (
                df.get_column(value_col_name)
                  .cast(pl.Utf8)
                  .str.split_exact(".", 1)
                  .struct.field("field_1")
                  .str.len_chars()
                  .fill_null(0)
                  .max()
            )
            if max_decimals is None:
                max_decimals = 0
            self.scale_factor = 10 ** max_decimals
        self._minSup_scaled = int(self.min_support_float * self.scale_factor)

        if self.debug:
            if self.forced_quant_mult is not None:
                print(f"Using forced quant_mult: {self.scale_factor}")
            else:
                print(f"Auto-detected max decimals: {max_decimals}")
                print(f"Calculated quant_mult (scale_factor): {self.scale_factor}")
            print(f"Scaled minimum support: {self._minSup_scaled}")

        # Apply scaling and select final columns
        df = df.with_columns(
            (pl.col(value_col_name).cast(pl.Float64) * self.scale_factor).cast(pl.UInt32).alias("prob_scaled")
        ).select(
            pl.col("item").cast(pl.Utf8),
            pl.col("prob_scaled").alias("prob"),
            pl.col("txn_id").cast(pl.UInt32)
        )

        
        return cudf.DataFrame.from_arrow(df.to_arrow())

    def _mine(self):
        """Orchestrates the main mining process.

        This method executes the end-to-end workflow of the algorithm:
        1.  Calls ``_cpu_load_and_scale_data`` to load and pre-process the data on the CPU.
        2.  Filters items that do not meet the minimum support threshold.
        3.  Allocates and builds the necessary hash table structures on the GPU.
        4.  Performs the iterative mining process on the GPU to find frequent itemsets.
        5.  Processes the results and cleans up GPU memory.
        """
        # --- 1. High-performance CPU load, parse, and scale ---
        exploded_gdf = self._cpu_load_and_scale_data()
        
        # --- 2. Filter Items and Setup for GPU (Unchanged) ---
        support_df, final_gdf = self._calculate_support_and_filter_items(exploded_gdf)
        if support_df is None:
            self.patterns = {}
            return
            
        # --- 3. Allocate and Build GPU Data Structures (Unchanged) ---
        ht_host = self._allocate_gpu_hash_tables(support_df)
        self._build_gpu_hash_tables(ht_host, final_gdf)

        # --- 4. Iterative Mining on GPU (Unchanged) ---
        k = 1
        candidates = self.support_df["new_id"].to_cupy().reshape(-1, 1)
        gpu_results = []
        while len(candidates) > 0:
            if self.debug: 
                print(f"Mining {k+1}-itemsets from {len(candidates)} candidates...")
            next_candidates = self._generate_next_candidates(candidates, k)
            if len(next_candidates) == 0: break
            k += 1
            supports = cp.zeros(len(next_candidates), dtype=cp.uint64)
            KERN_MINE((len(next_candidates),), (self._max_threads,), (self._ht_dev, next_candidates, len(next_candidates), k, supports))
            cp.cuda.runtime.deviceSynchronize()
            mask = supports >= self._minSup_scaled
            frequent_cands = next_candidates[mask]
            if len(frequent_cands) > 0: gpu_results.append((frequent_cands.get(), supports[mask].get()))
            candidates = frequent_cands

        # --- 5. Process Results and Cleanup ---
        self._process_results(gpu_results)
        self._gpu_memory_usage = self._get_gpu_memory_usage()
        del self.item_col, self.line_col, self.prob_col, self._ht_dev, self.buckets
        del self.support_df, self.rename_map
        cp.get_default_memory_pool().free_all_blocks()

    def _process_results(self, gpu_results):
        """Processes raw GPU results, decodes item names, and rescales supports.

        This method takes the raw output from the GPU mining phase, which consists of
        integer item IDs and scaled support counts. It converts the item IDs back to
        their original string names and rescales the support values back to their
        original floating-point representation.

        :param gpu_results: A list of tuples, where each tuple contains the frequent
                            candidate itemsets (as NumPy arrays of IDs) and their
                            corresponding scaled support counts.
        :type gpu_results: list
        """
        # Rescale the support of 1-itemsets
        patterns = {
            (item,): float(prob) / self.scale_factor
            for item, prob in zip(self.support_df["item"].to_numpy(), self.support_df["prob"].to_numpy())
        }
        # Rescale the support of k-itemsets (k > 1)
        for ids_cpu, vals in gpu_results:
            names = self.rename_map[ids_cpu]
            for name_tuple, sup in zip(names, vals):
                patterns[tuple(name_tuple)] = float(sup) / self.scale_factor
        self.patterns = patterns

    def _calculate_support_and_filter_items(self, gdf: cudf.DataFrame) -> Tuple[cudf.DataFrame, cudf.DataFrame]:
        """Calculates initial item supports, filters infrequent items, and remaps IDs.

        This function performs the first major filtering step. It groups the transaction
        data by item to calculate the total support for each. Items that do not meet
        the scaled minimum support threshold are discarded. The remaining frequent items
        are then assigned new, contiguous integer IDs for more efficient processing on the GPU.

        :param gdf: The input cuDF DataFrame containing columns 'item', 'prob', and 'txn_id'.
        :type gdf: cudf.DataFrame
        :return: A tuple containing:
                 - A cuDF DataFrame with support information for frequent items ('support_df').
                 - A new cuDF DataFrame with infrequent items removed and item names
                   replaced by their new integer IDs ('final_gdf').
                 Returns (None, None) if no frequent items are found.
        :rtype: Tuple[cudf.DataFrame, cudf.DataFrame]
        """
        support_df = gdf.groupby("item").agg({"prob": "sum", "txn_id": "count"}).rename(columns={"txn_id": "freq"})
        support_df = support_df[support_df["prob"] >= self._minSup_scaled].sort_values("freq", ascending=False).reset_index()
        if len(support_df) == 0: return None, None
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

    def _allocate_gpu_hash_tables(self, support_df: cudf.DataFrame) -> np.ndarray:
        """Allocates memory for hash tables on the GPU.

        Based on the frequency of each item, this method calculates the required size for
        each hash table on the GPU, applying a load factor to prevent collisions. It then
        allocates a single contiguous block of memory on the GPU to hold all hash tables
        and creates a host-side structure that contains pointers and metadata for these tables.

        :param support_df: A cuDF DataFrame containing the support and frequency of each item.
        :type support_df: cudf.DataFrame
        :return: A NumPy array of HashTable structures, configured for each item, ready to be
                 copied to the device.
        :rtype: np.ndarray
        """
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

    def _build_gpu_hash_tables(self, ht_host: np.ndarray, gdf: cudf.DataFrame):
        """Populates the GPU hash tables with transaction data.

        This method copies the host-side hash table structures to the GPU and then
        launches a CUDA kernel (``KERN_BUILD``) to insert all transaction data
        (item IDs, transaction IDs, and probabilities) into the appropriate hash tables
        on the device.

        :param ht_host: A NumPy array of HashTable structures created by
                        ``_allocate_gpu_hash_tables``.
        :type ht_host: np.ndarray
        :param gdf: The filtered and transformed cuDF DataFrame containing the transaction data.
        :type gdf: cudf.DataFrame
        """
        self.item_col = gdf["item"].astype("uint32").values
        self.line_col = gdf["txn_id"].astype("uint32").values
        self.prob_col = gdf["prob"].astype("uint64").values
        self._ht_dev = _to_device_struct(ht_host)
        total_pairs = len(self.item_col)
        grid_size = (total_pairs + self._max_threads - 1) // self._max_threads
        KERN_BUILD((grid_size,), (self._max_threads,), (total_pairs, self.item_col, self.line_col, self.prob_col, self._ht_dev))

    def _generate_next_candidates(self, cands: cp.ndarray, k: int) -> cp.ndarray:
        """Generates (k+1)-itemset candidates from frequent k-itemsets on the GPU.

        This function implements the candidate generation step of the Apriori algorithm.
        It takes a set of frequent k-itemsets and generates a new set of (k+1)-itemset
        candidates. The generation is performed entirely on the GPU using CUDA kernels
        for efficiency.

        :param cands: A CuPy array of frequent k-itemsets.
        :type cands: cp.ndarray
        :param k: The current itemset size.
        :type k: int
        :return: A new CuPy array of (k+1)-itemset candidates. Returns an empty array
                 if no new candidates can be generated.
        :rtype: cp.ndarray
        """
        n = len(cands)
        if n == 0: return cp.array([], dtype=cp.uint32)
        grid_size = (n + self._max_threads - 1) // self._max_threads
        num = cp.zeros(n + 1, dtype=cp.uint32)
        KERN_NUM((grid_size,), (self._max_threads,), (cands, n, k, num))
        idx = cp.cumsum(num, dtype=cp.uint32)
        out_len = int(idx[-1].get())
        if out_len == 0: return cp.array([], dtype=cp.uint32)
        nxt = cp.zeros((out_len, k + 1), dtype=cp.uint32)
        KERN_WRITE((grid_size,), (self._max_threads,), (cands, n, k, idx, nxt))
        return nxt

    def _get_gpu_memory_usage(self) -> int:
        """Reports the total GPU memory used by the algorithm.

        :return: The amount of GPU memory used, in bytes. Returns 0 if CUDA is not available
                 or if the query fails.
        :rtype: int
        """
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            return total - free
        except cp.cuda.runtime.CUDARuntimeError: return 0

    def save(self, oFile: Union[str, Path]):
        """Saves the discovered frequent patterns to a file.

        The patterns are written in a text format, with each line containing one
        itemset. The items in the set are comma-separated, followed by a tab,
        and then the floating-point support value.

        :param oFile: The path to the output file.
        :type oFile: Union[str, Path]
        """
        with open(oFile, "w") as f:
            for itemset, support in self.patterns.items(): f.write(f"{','.join(itemset)}\t{support:.6f}\n")

    def print_results(self):
        """Prints a summary of the mining results to the console.

        The summary includes execution time, peak CPU and GPU memory usage, and the
        total number of frequent patterns found.
        """
        print(f"\n--- {self.__class__.__name__} Results ---")
        print(f"Execution Time: {self.get_execution_time():.4f} seconds")
        print(f"Peak CPU Memory Usage: {self.get_memory_usage():.2f} MB")
        print(f"Peak GPU Memory Usage: {self._gpu_memory_usage / (1024**2):.2f} MB")
        print(f"Patterns Found: {self.get_pattern_count()}")
        print("--------------------" + "-" * len(self.__class__.__name__))

def main():
    """Main function to run the naiveFFIMiner algorithm from the command line.

    This function parses command-line arguments to configure and run the miner.
    It handles file I/O, instantiates the ``naiveFFIMiner`` class, and prints
    the final results.
    """
    parser = argparse.ArgumentParser(description="cuFFIMiner_Adv - A CUDA-based Fuzzy Frequent Itemset Miner with auto-scaling.")
    parser.add_argument("iFile", type=str, help="Path to the input dataset (.txt, .csv, or .parquet).")
    parser.add_argument("min_support", type=float, help="Minimum support threshold as a float (e.g., 0.05).")
    parser.add_argument("--quant-mult", type=int, default=None, help="Optional external quantization multiplier to force (bypass auto-detect).")
    parser.add_argument("-o", "--oFile", type=str, default="patterns.txt", help="Path to the output file.")
    parser.add_argument("--sep", type=str, default="\t", help="Separator for items in text input files.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    if not os.path.exists(args.iFile):
        print(f"Error: Input file not found at {args.iFile}")
        return
    
    algorithm = naiveFFIMiner(
        iFile=args.iFile,
        min_support=args.min_support,
        sep=args.sep,
        quant_mult=args.quant_mult,
        debug=args.debug
    )
    algorithm.mine()
    algorithm.save(args.oFile)
    algorithm.print_results()

if __name__ == "__main__":
    main()