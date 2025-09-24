// gds_bench_csr.cpp
// Refactored: functions, single big GDS transfer, print_head/print_summary integrated.
// Compile with: nvcc -std=c++17 -O3 -arch=native -Xcompiler "-fopenmp" \
//   -o gds_bench_csr gds_bench_csr.cpp -lcufile -lboost_program_options -lnvToolsExt

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cerrno>
#include <iomanip>
#include <memory>
#include <numeric>
#include <atomic>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/resource.h>

#include <cuda_runtime.h>
#include <cufile.h>
#include <nvtx3/nvToolsExt.h>

#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;
using clk = chrono::high_resolution_clock;

// -----------------------------
// Error checking macros
// -----------------------------
#define CUDA_CHECK(call)                                                   \
    do                                                                     \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess)                                            \
        {                                                                  \
            cerr << "CUDA Error in " << #call << " at " << __FILE__ << ":" \
                 << __LINE__ << ": " << cudaGetErrorString(err) << endl;   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define GDS_CHECK(call)                                                         \
    do                                                                          \
    {                                                                           \
        CUfileError_t err = call;                                               \
        if (err.err != CU_FILE_SUCCESS)                                         \
        {                                                                       \
            cerr << "GDS Error in " << #call << " at " << __FILE__ << ":"       \
                 << __LINE__ << ": " << cufileop_status_error(err.err) << endl; \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// -----------------------------
// Utilities
// -----------------------------
static inline size_t round_up(size_t x, size_t a)
{
    return ((x + a - 1) / a) * a;
}

struct Options
{
    string filename;
    int io_size_kb = 1024;
    int block_size_kb = 128; // CUDA kernel processing block size (unused for read)
    char delim = ',';
    bool odirect = true;
};

// -----------------------------
// CUDA Kernels (unchanged logic)
// -----------------------------
__global__ void parse_and_count_lines_kernel(const char *data, size_t data_size, char delim,
                                             uint64_t *total_numbers_gpu, uint64_t *line_counts_gpu,
                                             size_t max_lines)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t current_line = 0;

    __shared__ size_t line_count_shared;
    __shared__ size_t total_numbers_shared;
    if (threadIdx.x == 0){
        line_count_shared = 0;
        total_numbers_shared = 0;
    }
    __syncthreads();

    for (size_t i = idx; i < data_size; i += stride)
    {
        if (i > 0 && data[i - 1] == '\n')
        {
            current_line++;
        }

        bool is_start_of_num = (data[i] >= '0' && data[i] <= '9') &&
                               (i == 0 || data[i - 1] == delim || data[i - 1] == '\n');

        if (is_start_of_num)
        {
            // atomicAdd(reinterpret_cast<unsigned long long *>(total_numbers_gpu), 1ULL);
            atomicAdd(reinterpret_cast<unsigned long long *>(&total_numbers_shared), 1ULL);
            if (current_line < max_lines)
            {
                // atomicAdd(reinterpret_cast<unsigned long long *>(&line_counts_gpu[current_line]), 1ULL);
                atomicAdd(reinterpret_cast<unsigned long long *>(&line_count_shared), 1ULL);
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0){
        atomicAdd(reinterpret_cast<unsigned long long *>(total_numbers_gpu), static_cast<unsigned long long>(total_numbers_shared));
        if (current_line < max_lines){
            atomicAdd(reinterpret_cast<unsigned long long *>(&line_counts_gpu[current_line]), static_cast<unsigned long long>(line_count_shared));
        }
    }
}

__global__ void prefix_sum_to_csr_kernel(const uint64_t *line_counts, uint64_t *row_ptrs, size_t num_lines)
{
    // Simple single-threaded kernel: run on 1 block/1 thread
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        row_ptrs[0] = 0;
        uint64_t current_sum = 0;
        for (size_t i = 0; i < num_lines; ++i)
        {
            current_sum += line_counts[i];
            row_ptrs[i + 1] = current_sum;
        }
    }
}

// -----------------------------
// Forward declarations
// -----------------------------
bool parse_cli(int argc, char **argv, Options &opt);
size_t get_line_count(const string &filename);
int open_file(const string &filename, bool odirect);
bool get_file_info(int fd, size_t &file_size, size_t &fs_block);
void *allocate_aligned_host_buffer(size_t size, size_t alignment);
bool init_gds_driver();
bool register_buffer_with_gds(void *buf, size_t size);
ssize_t single_big_gds_read(CUfileHandle_t cf_handle, char *d_buffer, size_t file_size);
void print_head(const void *buffer, size_t size);
void print_summary(size_t file_size, double read_seconds, uint64_t parsed_numbers, double parse_seconds, bool odirect, size_t io_errors);
pair<uint64_t, double> run_parse_stage(char *d_buffer, size_t file_size, size_t num_lines, char delim);

// -----------------------------
// Implementations
// -----------------------------
bool parse_cli(int argc, char **argv, Options &opt)
{
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "show help")("file", po::value<string>()->required(), "input file (positional)")("io-size-kb,i", po::value<int>()->default_value(1024), "GDS IO size per request (KB)")("delim,d", po::value<char>()->default_value(','), "delimiter for parse stage")("no-odirect", "disable O_DIRECT");

    po::positional_options_description p;
    p.add("file", 1);
    po::variables_map vm;
    try
    {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
        if (vm.count("help"))
        {
            cout << "gds_bench_csr - GDS read + CUDA parse benchmark\n\n";
            cout << "Usage: " << argv[0] << " <file> [options]\n\n"
                 << desc << "\n";
            return false;
        }
        po::notify(vm);
    }
    catch (const po::error &e)
    {
        cerr << "Error parsing options: " << e.what() << "\n\n"
             << desc << "\n";
        return false;
    }
    opt.filename = vm["file"].as<string>();
    opt.io_size_kb = vm["io-size-kb"].as<int>();
    opt.delim = vm["delim"].as<char>();
    opt.odirect = (vm.count("no-odirect") == 0);
    return true;
}

size_t get_line_count(const string &filename)
{
    FILE *f = fopen(filename.c_str(), "r");
    if (!f)
        return 0;
    size_t count = 0;
    const size_t BUF_SZ = 64 * 1024;
    vector<char> buf(BUF_SZ);
    while (!feof(f))
    {
        size_t bytes_read = fread(buf.data(), 1, buf.size(), f);
        for (size_t i = 0; i < bytes_read; ++i)
        {
            if (buf[i] == '\n')
                count++;
        }
        if (bytes_read == 0)
            break;
    }
    fclose(f);
    // Add one for the last line if the file doesn't end with a newline
    struct stat st;
    if (stat(filename.c_str(), &st) == 0 && st.st_size > 0)
        count++;
    return count;
}

int open_file(const string &filename, bool odirect)
{
    int flags = O_RDONLY;
    if (odirect)
        flags |= O_DIRECT;
    int fd = open(filename.c_str(), flags);
    return fd;
}

bool get_file_info(int fd, size_t &file_size, size_t &fs_block)
{
    struct stat st;
    if (fstat(fd, &st) != 0)
    {
        perror("fstat");
        return false;
    }
    file_size = static_cast<size_t>(st.st_size);

    // try to obtain filesystem block size / IO alignment
    fs_block = static_cast<size_t>(st.st_blksize ? st.st_blksize : 4096);
    return true;
}

void *allocate_aligned_host_buffer(size_t size, size_t alignment)
{
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0)
        return nullptr;
    memset(ptr, 0, size);
    return ptr;
}

bool init_gds_driver()
{
    CUfileError_t e = cuFileDriverOpen();
    if (e.err != CU_FILE_SUCCESS)
    {
        cerr << "cuFileDriverOpen failed: " << cufileop_status_error(e.err) << endl;
        return false;
    }
    return true;
}

bool register_buffer_with_gds(void *buf, size_t size)
{
    CUfileError_t e = cuFileBufRegister(buf, size, 0);
    if (e.err != CU_FILE_SUCCESS)
    {
        cerr << "cuFileBufRegister failed: " << cufileop_status_error(e.err) << endl;
        return false;
    }
    return true;
}

/**
 * Attempt to read the entire file with a single cuFileRead call.
 * If cuFileRead returns a partial read, fall back to reading remaining bytes in a loop.
 * Returns total bytes read or negative on fatal error.
 */
ssize_t single_big_gds_read(CUfileHandle_t cf_handle, char *d_buffer, size_t file_size)
{
    ssize_t total_read = 0;
    ssize_t ret = cuFileRead(cf_handle, d_buffer, file_size, 0, 0);
    if (ret < 0)
    {
        // fatal read error
        return ret;
    }
    total_read += ret;

    // If partial, continue reading remaining bytes (should be rare with GDS single large read).
    while (static_cast<size_t>(total_read) < file_size)
    {
        ssize_t rem = static_cast<ssize_t>(file_size - static_cast<size_t>(total_read));
        ssize_t r = cuFileRead(cf_handle, d_buffer + total_read, rem, total_read, 0);
        if (r < 0)
        {
            return r; // error
        }
        total_read += r;
    }
    return total_read;
}

/**
 * @brief Prints the first two lines of a buffer for verification.
 * @param buffer Pointer to the data.
 * @param size The total size of the data.
 */
void print_head(const void *buffer, size_t size)
{
    const char *data = static_cast<const char *>(buffer);
    size_t current_pos = 0;
    int newlines_found = 0;
    size_t print_limit = std::min(size, (size_t)1024); // Cap max output to prevent huge dumps

    cout << "\n--- FILE HEAD (first 2 lines) ---\n";

    while (current_pos < print_limit && newlines_found < 2)
    {
        cout.put(data[current_pos]);
        if (data[current_pos] == '\n')
        {
            newlines_found++;
        }
        current_pos++;
    }

    if (current_pos < size && newlines_found < 2) // If we hit print_limit before 2 newlines
    {
        cout << "[...]\n";
    }
    else if (current_pos < size) // If we have more content beyond 2 newlines
    {
        if (current_pos > 0 && data[current_pos - 1] != '\n')
        {
            cout << "\n";
        }
        cout << "[...]\n";
    }
    cout << "[...]\n";

}

/**
 * @brief Prints a final summary of the benchmark results.
 * @param file_size Total size of the file.
 * @param read_seconds Time taken for the read stage.
 * @param parsed_numbers Total numbers counted in the parse stage.
 * @param parse_seconds Time taken for the parse stage.
 * @param odirect Whether O_DIRECT was used.
 * @param io_errors The number of failed I/O requests.
 */
void print_summary(size_t file_size, double read_seconds, uint64_t parsed_numbers, double parse_seconds, bool odirect, size_t io_errors)
{
    double file_mb = double(file_size) / (1024.0 * 1024.0);
    cout << fixed << setprecision(3);
    cout << "\n==== BENCHMARK SUMMARY ====\n";
    cout << "O_DIRECT used: " << (odirect ? "YES" : "NO") << "\n";
    cout << "File size: " << file_size << " bytes (" << file_mb << " MB)\n";
    cout << "\n-- READ STAGE --\n";
    cout << "Failed I/O requests: " << io_errors << "\n";
    cout << "Elapsed read time: " << read_seconds << " s\n";
    if (io_errors == 0)
    {
        cout << "Read throughput: " << (file_mb / max(read_seconds, 1e-12)) << " MB/s\n";
    }
    else
    {
        cout << "Read throughput: N/A (due to I/O errors)\n";
    }
    cout << "\n-- PARSE STAGE --\n";
    cout << "Parsed numeric tokens: " << parsed_numbers << "\n";
    cout << "Parse elapsed: " << parse_seconds << " s\n";
    if (io_errors == 0)
    {
        cout << "Parse throughput (fileMB/parse_time): " << (file_mb / max(parse_seconds, 1e-12)) << " MB/s\n";
    }
    else
    {
        cout << "Parse throughput: N/A (due to I/O errors and potentially corrupt data)\n";
    }

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    cout << "\nPeak RSS: " << ru.ru_maxrss << " KB\n";
    cout << "===========================\n";
}

/**
 * Run the CUDA parse kernels: counting numeric tokens and generating CSR row_ptrs.
 * Returns pair<total_numbers, parse_time_seconds>.
 */
pair<uint64_t, double> run_parse_stage(char *d_buffer, size_t file_size, size_t num_lines, char delim)
{
    nvtxRangePushA("CUDA Parse");
    auto t0_parse = clk::now();

    // Allocate GPU memory for counters
    uint64_t *d_total_numbers = nullptr;
    uint64_t *d_line_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_total_numbers, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_line_counts, sizeof(uint64_t) * max((size_t)1, num_lines)));
    CUDA_CHECK(cudaMemset(d_total_numbers, 0, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_line_counts, 0, sizeof(uint64_t) * max((size_t)1, num_lines)));

    // Kernel launch configuration
    int blockSize = 1024; // threads per block
    // choose gridSize based on file_size bytes to give enough parallelism
    int gridSize = static_cast<int>((file_size + blockSize - 1) / blockSize);
    // cap grid size to a reasonable number to avoid oversubscription
    if (gridSize < 1) gridSize = 1;
    if (gridSize > 65535) gridSize = 65535;

    parse_and_count_lines_kernel<<<gridSize, blockSize>>>(d_buffer, file_size, delim,
                                                          d_total_numbers, d_line_counts, num_lines);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1_parse_kernels = clk::now();

    // CSR generation
    nvtxRangePushA("CSR Generation");
    uint64_t *d_row_ptrs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_row_ptrs, sizeof(uint64_t) * (max((size_t)1, num_lines) + 1)));
    prefix_sum_to_csr_kernel<<<1, 1>>>(d_line_counts, d_row_ptrs, num_lines);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop(); // CSR Generation

    auto t1_parse = clk::now();
    double parse_seconds = chrono::duration<double>(t1_parse - t0_parse).count();

    // Copy totals back
    uint64_t total_numbers_h = 0;
    CUDA_CHECK(cudaMemcpy(&total_numbers_h, d_total_numbers, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // cleanup GPU allocations used in parse
    CUDA_CHECK(cudaFree(d_total_numbers));
    CUDA_CHECK(cudaFree(d_line_counts));
    CUDA_CHECK(cudaFree(d_row_ptrs));

    nvtxRangePop(); // CUDA Parse
    return {total_numbers_h, parse_seconds};
}

// -----------------------------
// Main
// -----------------------------
int main(int argc, char **argv)
{
    Options opt;
    if (!parse_cli(argc, argv, opt))
        return 1;

    nvtxRangePushA("Setup");

    // Open file
    int fd = open_file(opt.filename, opt.odirect);
    if (fd < 0)
    {
        perror("open");
        return 1;
    }

    size_t file_size = 0;
    size_t fs_block = 0;
    if (!get_file_info(fd, file_size, fs_block))
    {
        close(fd);
        return 1;
    }
    if (file_size == 0)
    {
        cerr << "Empty file.\n";
        close(fd);
        return 0;
    }

    size_t io_chunk = static_cast<size_t>(opt.io_size_kb) * 1024;
    // for registration and O_DIRECT alignment, align to filesystem block size
    size_t alloc_size = round_up(file_size, fs_block);
    size_t registration_size = round_up(alloc_size, io_chunk);

    cout << "Starting GDS benchmark\n";
    cout << "File: " << opt.filename << " size: " << file_size << " bytes\n";
    cout << "Requested IO chunk size: " << opt.io_size_kb << " KB\n";
    cout << "Alignment / FS block: " << fs_block << " bytes\n";
    cout << "Alloc/register size (rounded): " << registration_size << " bytes\n";

    // Initialize GDS
    if (!init_gds_driver())
    {
        close(fd);
        return 1;
    }

    // Prepare cuFile descriptor & register handle
    CUfileDescr_t cf_desc;
    memset(&cf_desc, 0, sizeof(CUfileDescr_t));
    cf_desc.handle.fd = fd;
    cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileHandle_t cf_handle;
    GDS_CHECK(cuFileHandleRegister(&cf_handle, &cf_desc));

    // Allocate device buffer and register with GDS
    char *d_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buffer, registration_size));
    GDS_CHECK(cuFileBufRegister(d_buffer, registration_size, 0));

    nvtxRangePop(); // Setup

    // Perform single big GDS read
    nvtxRangePushA("GDS Transfer");
    auto t0_transfer = clk::now();

    ssize_t bytes_read = single_big_gds_read(cf_handle, d_buffer, file_size);
    if (bytes_read < 0)
    {
        cerr << "GDS read failed with error: " << bytes_read << "\n";
        // cleanup & exit
        GDS_CHECK(cuFileBufDeregister(d_buffer));
        cuFileHandleDeregister(cf_handle);
        GDS_CHECK(cuFileDriverClose());
        CUDA_CHECK(cudaFree(d_buffer));
        close(fd);
        return 1;
    }

    // Ensure device synchronization
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1_transfer = clk::now();
    double transfer_seconds = chrono::duration<double>(t1_transfer - t0_transfer).count();
    nvtxRangePop(); // GDS Transfer

    // Pre-scan for line count (we already had a function)
    size_t num_lines = get_line_count(opt.filename);
    cout << "Pre-scanned file, found ~" << num_lines << " lines.\n";

    // Run parse stage
    auto parse_result = run_parse_stage(d_buffer, file_size, num_lines, opt.delim);

    // Copy a small host slice for print_head (we'll copy up to 1KB or file_size)
    size_t head_copy_size = min(file_size, (size_t)1024);
    vector<char> host_head(head_copy_size);
    CUDA_CHECK(cudaMemcpy(host_head.data(), d_buffer, head_copy_size, cudaMemcpyDeviceToHost));
    print_head(host_head.data(), head_copy_size);

    // Print final summary
    size_t io_errors = 0; // we only do a single read attempt with fallback loop; count errors if ret < 0
    print_summary(file_size, transfer_seconds, parse_result.first, parse_result.second, opt.odirect, io_errors);

    // Cleanup
    nvtxRangePushA("Cleanup");
    GDS_CHECK(cuFileBufDeregister(d_buffer));
    cuFileHandleDeregister(cf_handle);
    GDS_CHECK(cuFileDriverClose());
    close(fd);
    CUDA_CHECK(cudaFree(d_buffer));
    nvtxRangePop(); // Cleanup

    return 0;
}
