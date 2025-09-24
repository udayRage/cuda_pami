# CSV Throughput Benchmark

This tool benchmarks raw SSD read speeds and CPU parsing throughput
for CSV-like integer data. It uses:
- **O_DIRECT + io_uring** for low-level async disk reads
- **OpenMP** for parallel parsing on CPU
- **Boost.Program_options** for easy CLI handling

## Build

```bash
mkdir build && cd build
cmake ..
make -j
