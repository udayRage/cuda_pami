"""
cuFFIMiner - A CUDA-based Fuzzy Frequent Itemset Miner
======================================================

This module implements the cuFFIMiner algorithm, which leverages CUDA for high-performance
fuzzy frequent itemset mining. It is designed to work with the BaseAlgorithm framework.

It assumes the input data contains **pre-scaled uint32 integers** for probabilities.

Input Formats:
---------------
1.  **Text Files (.csv, .txt, .tsv)**: Transactional format, where each line contains
    items and their corresponding integer values, separated by a special character.
    Example: `itemA<sep>itemB:100<sep>80`

2.  **Parquet Files (.parquet)**: Pre-processed (long/tidy) format. The file must
    contain the columns `item` (str), `prob` (uint32), and `txn_id` (uint32). This
    format is significantly faster to load as it bypasses text parsing.

"""

import os
import argparse
from pathlib import Path
from typing import Union, Dict, Tuple
import re

# Import everything from the base module.
from ..base import *

# Check for GPU availability and necessary libraries
try:
    import cupy as cp
    import cudf
    import pandas as pd
    GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

# --- CUDA & Host Data Structures (unchanged) ---
THREADS = 1024
INV_LOAD_FACTOR = 3/2
KVPair = np.dtype([("line", np.uint32), ("probability", np.uint64)], align=True)
HashTable = np.dtype([("table_ptr", np.uintp), ("table_size", np.uint32), ("mask", np.uint32), ("id", np.uint32)], align=True)
assert KVPair.itemsize == 16, "Host KVPair layout must be 16 Bytes"
assert HashTable.itemsize == 24, "Host HashTable layout must be 24 Bytes"

def _to_device_struct(arr: np.ndarray) -> cp.ndarray:
    """Helper to correctly cast NumPy structured array to a CuPy view for kernel launch."""
    return cp.asarray(arr.view(np.uint8)).view(arr.dtype)

SENTINEL = 0
FIRST_ID = 1

# --- CUDA Kernel Source Code (unchanged) ---
KERNEL_SRC = r"""
// ... (CUDA kernel code remains the same as the previous version) ...
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

    RAWMOD = cp.RawModule(code=KERNEL_SRC, backend="nvcc", options=tuple(NVCC_OPTIONS))
    KERN_NUM = RAWMOD.get_function("number_of_new_candidates_to_generate")
    KERN_WRITE = RAWMOD.get_function("write_the_new_candidates")
    KERN_BUILD = RAWMOD.get_function("build_tables")
    KERN_MINE = RAWMOD.get_function("mine_kernel")
else:
    RAWMOD = KERN_NUM = KERN_WRITE = KERN_BUILD = KERN_MINE = None

class cuFFIMiner(BaseAlgorithm):
    """
    Implementation of the cuFFIMiner algorithm using the BaseAlgorithm framework.
    It assumes the input data contains pre-scaled integer probabilities.

    :ivar patterns: A dictionary storing the discovered frequent patterns.
                    The keys are tuples of item names, and the values are their fuzzy support.
    """

    def __init__(self, iFile: Union[str, Path, pd.DataFrame], min_support: int, sep: str = "\t", scaling_factor: int = 1, memory_type: str = "global", debug: bool = False):
        """
        Initializes the cuFFIMiner algorithm.

        :param iFile: Input file path (text or Parquet).
        :type iFile: Union[str, Path]
        :param min_support: The minimum support threshold as a scaled integer (e.g., 100).
        :type min_support: int
        :param sep: The separator used in text input files.
        :type sep: str
        :param scaling_factor: The factor used to scale probabilities from fixed-point to floating-point.
        :type scaling_factor: int
        :param memory_type: The type of GPU memory to use ('global', 'pinned', 'unified').
        :type memory_type: str
        :param debug: If True, enables detailed memory and CPU profiling.
        :type debug: bool
        """
        # check if file is there
        if not os.path.exists(iFile):
            raise FileNotFoundError(f"Input file not found: {iFile}")

        super().__init__(iFile=iFile, debug=debug)
        if not GPU_AVAILABLE:
            raise RuntimeError("cuFFIMiner requires a GPU with cuDF and CuPy installed.")

        self._dpool = None
        self._ppool = None
        self._configure_memory_pool(memory_type)

        self._minSup_scaled = min_support
        self._scaling_factor = scaling_factor
        self._sep = sep
        self._gpu_memory_usage = 0
        self.patterns: Dict[Tuple[str, ...], float] = {}

    def _configure_memory_pool(self, memory_type: str):
        """
        Configure device allocations, and (optionally) enable a pinned host pool.
        memory_type: 'global' | 'unified' | 'global+pinned' | 'unified+pinned'
        """
        use_pinned = memory_type.endswith("+pinned")
        base = memory_type.replace("+pinned", "")

        if base == "global":
            self._dpool = cp.cuda.MemoryPool()                       # device allocations
            cp.cuda.set_allocator(self._dpool.malloc)
        elif base == "unified":
            self._dpool = cp.cuda.MemoryPool(cp.cuda.malloc_managed) # device allocations (UM)
            cp.cuda.set_allocator(self._dpool.malloc)
        else:
            raise ValueError(f"Invalid memory type '{memory_type}'. "
                            "Use 'global', 'unified', optionally with '+pinned'.")

        if use_pinned:
            self._ppool = cp.cuda.PinnedMemoryPool()                 # host pinned allocations
            cp.cuda.set_pinned_memory_allocator(self._ppool.malloc)

    def _load_data(self) -> cudf.DataFrame:
        """
        Loads transactional data from a text file into a cuDF DataFrame.

        :return: A cuDF DataFrame with list columns for 'items' and 'values'.
        :rtype: cudf.DataFrame
        """
        gdf = cudf.read_csv(str(self._iFile), sep=":", header=None, names=["items", "values"], dtype=["str", "str"])
        sep_esc = re.escape(self._sep)
        pattern = rf"[{sep_esc}\r\n ]+$"
        gdf["items"] = gdf["items"].str.replace(pattern, "", regex=True)
        gdf["values"] = gdf["values"].str.replace(pattern, "", regex=True)
        gdf["items"] = gdf["items"].str.split(self._sep)
        gdf["values"] = gdf["values"].str.split(self._sep)
        gdf = gdf.reset_index(drop=True)
        gdf["txn_id"] = (gdf.index + 1).astype("uint32")
        return gdf
    
    def _load_parquet_data(self) -> cudf.DataFrame:
        """
        Loads pre-processed data from a Parquet file and validates its structure.

        The Parquet file must contain the following columns:
        - `item` (string): The name of the item.
        - `prob` (uint32): The pre-scaled integer probability/value.
        - `txn_id` (uint32): The transaction identifier.

        :raises ValueError: If the Parquet file is missing required columns or if
                            columns have incorrect data types.
        :return: A validated cuDF DataFrame in the required long format.
        :rtype: cudf.DataFrame
        """
        gdf = cudf.read_parquet(str(self._iFile))
        
        REQUIRED_COLUMNS = ["item", "prob", "txn_id"]
        if not all(col in gdf.columns for col in REQUIRED_COLUMNS):
            raise ValueError(f"Parquet file must contain the columns: {REQUIRED_COLUMNS}")
            
        try:
            gdf['item'] = gdf['item'].astype('str')
            gdf['prob'] = gdf['prob'].astype('uint32')
            gdf['txn_id'] = gdf['txn_id'].astype('uint32')
        except (TypeError, ValueError) as e:
            raise ValueError("Parquet file columns could not be cast to required types (item:str, prob:uint32, txn_id:uint32).") from e

        return gdf

    def _explode_transactions(self, gdf: cudf.DataFrame) -> cudf.DataFrame:
        """
        Converts a DataFrame from transactional format (lists) to long format.

        :param gdf: The input DataFrame with list columns.
        :type gdf: cudf.DataFrame
        :raises ValueError: If values cannot be converted to uint32.
        :return: An exploded DataFrame with columns ['item', 'prob', 'txn_id'].
        :rtype: cudf.DataFrame
        """
        items_ex = gdf[["txn_id", "items"]].explode("items")
        values_ex = gdf[["txn_id", "values"]].explode("values")
        items_ex["pos"] = items_ex.groupby("txn_id").cumcount()
        values_ex["pos"] = values_ex.groupby("txn_id").cumcount()
        gdf = items_ex.merge(values_ex, on=["txn_id", "pos"]).drop(columns="pos")
        
        try:
            gdf["prob"] = gdf["values"].astype("uint32")
        except (ValueError, TypeError) as e:
            raise ValueError("Values in text file must be convertible to uint32.") from e

        return gdf.drop(columns="values").rename(columns={"items": "item"})

    def _calculate_support_and_filter_items(self, gdf: cudf.DataFrame) -> Tuple[cudf.DataFrame, cudf.DataFrame]:
        """
        Calculates support, filters infrequent items, and assigns new dense integer IDs.

        :param gdf: The exploded DataFrame.
        :type gdf: cudf.DataFrame
        :return: A tuple containing:
                 - The support DataFrame for frequent items.
                 - The main DataFrame with infrequent items removed and IDs updated.
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
        self.item_col = gdf["item"].astype("uint32").values
        self.line_col = gdf["txn_id"].astype("uint32").values
        self.prob_col = gdf["prob"].astype("uint64").values
        self._ht_dev = _to_device_struct(ht_host)
        total_pairs = len(self.item_col)
        grid_size = (total_pairs + self._max_threads - 1) // self._max_threads
        KERN_BUILD((grid_size,), (self._max_threads,),
                   (total_pairs, self.item_col, self.line_col, self.prob_col, self._ht_dev))

    def _mine(self):
        """
        Orchestrates the main mining process, handling both Parquet and text inputs.
        """
        is_parquet = isinstance(self._iFile, (str, Path)) and str(self._iFile).lower().endswith(".parquet")
        
        if is_parquet:
            exploded_gdf = self._load_parquet_data()
        else:
            raw_gdf = self._load_data()
            exploded_gdf = self._explode_transactions(raw_gdf)
        
        support_df, final_gdf = self._calculate_support_and_filter_items(exploded_gdf)
        if support_df is None:
            self.patterns = {}
            return
            
        ht_host = self._allocate_gpu_hash_tables(support_df)
        self._build_gpu_hash_tables(ht_host, final_gdf)

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
            if len(frequent_cands) > 0: 
                gpu_results.append(
                    (frequent_cands.get(), supports[mask].get())
                )
            candidates = frequent_cands

        self._process_results(gpu_results)
        self._gpu_memory_usage = self._get_gpu_memory_usage()
        del self.item_col, self.line_col, self.prob_col, self._ht_dev, self.buckets
        del self.support_df, self.rename_map
        
        if self._dpool:
            self._dpool.free_all_blocks()
        if self._ppool:
            self._ppool.free_all_blocks()
        
        # Reset to default allocator
        cp.cuda.set_allocator(None)
        cp.cuda.set_pinned_memory_allocator(None)

    def _generate_next_candidates(self, cands: cp.ndarray, k: int) -> cp.ndarray:
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

    def _process_results(self, gpu_results):
        patterns = {(item,): float(prob) for item, prob in zip(self.support_df["item"].to_numpy(), self.support_df["prob"].to_numpy() / self._scaling_factor)}
        for ids_cpu, vals in gpu_results:
            # ensure supports are floating-point and avoid integer division issues
            vals = np.asarray(vals).astype(np.float64) / float(self._scaling_factor)
            names = self.rename_map[ids_cpu]
            for name_tuple, sup in zip(names, vals):
                patterns[tuple(name_tuple)] = float(sup)
        self.patterns = patterns

    def _get_gpu_memory_usage(self) -> int:
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            return total - free
        except cp.cuda.runtime.CUDARuntimeError: return 0

    def save(self, oFile: Union[str, Path]):
        with open(oFile, "w") as f:
            for itemset, support in self.patterns.items(): f.write(f"{','.join(itemset)}\t{support}\n")

    def print_results(self):
        print(f"\n--- {self.__class__.__name__} Results ---")
        print(f"Execution Time: {self.get_execution_time():.4f} seconds")
        print(f"Peak CPU Memory Usage: {self.get_memory_usage():.2f} MB")
        print(f"Peak GPU Memory Usage: {self._gpu_memory_usage / (1024**2):.2f} MB")
        print(f"Patterns Found: {self.get_pattern_count()}")
        print("--------------------" + "-" * len(self.__class__.__name__))

def main():
    """Main function to run the cuFFIMiner algorithm from the command line."""
    parser = argparse.ArgumentParser(description="cuFFIMiner - A CUDA-based Fuzzy Frequent Itemset Miner for pre-scaled integer data.")
    parser.add_argument("iFile", type=str, help="Path to the input dataset (.txt, .csv, or .parquet).")
    parser.add_argument("min_support", type=int, help="Minimum support threshold as a scaled integer (e.g., 100).")
    parser.add_argument("scaling_factor", type=int, help="Scaling factor for the input data. Will be used to scale down the output, fixed->floating")
    parser.add_argument("-o", "--oFile", type=str, default="patterns.txt", help="Path to the output file.")
    parser.add_argument("--sep", type=str, default="\t", help="Separator for items in text input files.")
    parser.add_argument("--mem_type", type=str, default="global", choices=["global", "pinned", "unified"], help="Type of GPU memory to use (global, pinned, unified).")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    if not os.path.exists(args.iFile):
        print(f"Error: Input file not found at {args.iFile}")
        return
    
    algorithm = cuFFIMiner(
        iFile=args.iFile,
        min_support=args.min_support,
        sep=args.sep,
        scaling_factor=args.scaling_factor,
        memory_type=args.mem_type,
        debug=args.debug
    )
    algorithm.mine()
    algorithm.save(args.oFile)
    algorithm.print_results()

if __name__ == "__main__":
    main()