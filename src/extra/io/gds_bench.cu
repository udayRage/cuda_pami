// gds_bench_fixed_parse.cu
// GDS read with multi-stage GPU parallel parse.
// Compile with:
// nvcc -std=c++17 -O3 -arch=native -Xcompiler "-fopenmp" -o gds_bench gds_bench_fixed_parse.cu -lcufile -lboost_program_options
//

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cerrno>
#include <iomanip>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/resource.h>

#include <cuda_runtime.h>
#include <cufile.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;
using clk = chrono::high_resolution_clock;

// -----------------------------
// Error checking macros
// -----------------------------
#define CUDA_CHECK(call)                                                   \
    do                                                                      \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            std::cerr << "CUDA Error: " << #call << " at " << __FILE__      \
                      << ":" << __LINE__ << " : " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define GDS_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        CUfileError_t _gds_err = call;                                      \
        if (_gds_err.err != CU_FILE_SUCCESS)                                \
        {                                                                    \
            std::cerr << "GDS Error: " << #call << " at " << __FILE__       \
                      << ":" << __LINE__ << " : " << cufileop_status_error(_gds_err.err) << std::endl; \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
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
    char delim = ',';
    bool odirect = true;
};

// Custom struct to hold all results from the parse stage
struct ParseResult
{
    unsigned int *d_final_data = nullptr; // Device pointer to the final parsed integer data
    uint64_t total_lines = 0;
    uint64_t total_items = 0;
    double parse_seconds = 0.0;
};


// -----------------------------
// CUDA Kernels for multi-stage parsing
// -----------------------------

// Kernel 1: Count the number of newline characters
__global__ void num_new_lines_kernel(const char *data, size_t size, unsigned int *numLines)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    if (data[tid] == '\n') {
        atomicAdd(&numLines[0], 1);
    }
}

// Kernel 2: Find the indices of all newline characters
__global__ void find_new_lines_kernel(const char *data, size_t size, unsigned long long *newline_indices)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    if (data[tid] == '\n')
    {
        unsigned long long index = atomicAdd(&newline_indices[0], 1);
        newline_indices[index + 1] = tid;
    }
}

// Kernel 3: Count the number of items per line
__global__ void get_items_per_line_kernel(const char *data, const unsigned long long *indexes, size_t numLines, int *items_per_line, char delimiter)
{
    size_t lineIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (lineIdx >= numLines) return;

    size_t start = (lineIdx == 0) ? 0 : indexes[lineIdx] + 1;
    size_t end = indexes[lineIdx + 1];

    int local_items = 1; // Start with 1 item
    for (size_t i = start; i < end; i++)
    {
        if (data[i] == delimiter)
        {
            local_items++;
        }
    }
    items_per_line[lineIdx] = local_items;
}


// Device-side atoi for the final conversion kernel
__device__ int my_atoi(const char *str, int len)
{
    int result = 0;
    int sign = 1;
    int i = 0;

    // Skip leading spaces
    while (i < len && str[i] == ' ') {
        i++;
    }

    // Handle sign
    if (i < len && str[i] == '-') {
        sign = -1;
        i++;
    } else if (i < len && str[i] == '+') {
        i++;
    }

    // Convert digits
    while (i < len && str[i] >= '0' && str[i] <= '9') {
        result = result * 10 + (str[i] - '0');
        i++;
    }

    return sign * result;
}


// Kernel 4: Convert character data to integers
__global__ void convert_char_to_int_kernel(const char *data, const unsigned long long *line_start_indices, const unsigned long long *item_offsets, size_t numLines, unsigned int *rawData, char separator)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numLines) return;

    size_t line_start = (tid == 0) ? 0 : line_start_indices[tid] + 1;
    size_t line_end = line_start_indices[tid + 1];

    char buffer[32]; // Max length for a 32-bit integer string
    int bufferIndex = 0;
    unsigned long long write_idx = item_offsets[tid];

    for (size_t k = line_start; k <= line_end; k++)
    {
        if (k == line_end || data[k] == separator)
        {
            if (bufferIndex > 0)
            {
                rawData[write_idx++] = my_atoi(buffer, bufferIndex);
                bufferIndex = 0;
            }
        }
        else
        {
            if (bufferIndex < sizeof(buffer))
            {
                buffer[bufferIndex++] = data[k];
            }
        }
    }
}


// -----------------------------
// Forward declarations
// -----------------------------
bool parse_cli(int argc, char **argv, Options &opt);
int open_file(const string &filename, bool odirect);
bool get_file_info(int fd, size_t &file_size, size_t &fs_block);
bool init_gds_driver();
ssize_t single_big_gds_read(CUfileHandle_t cf_handle, char *d_buffer, size_t file_size);
void print_head(const void *buffer, size_t size);
void print_summary(size_t file_size, double read_seconds, const ParseResult& result, bool odirect);
ParseResult run_parse_stage(char *d_buffer, size_t file_size, char delim);

// -----------------------------
// Implementations
// -----------------------------
bool parse_cli(int argc, char **argv, Options &opt)
{
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "show help")("file", po::value<string>()->required(), "input file (positional)")("delim,d", po::value<char>()->default_value(','), "delimiter for parse stage")("no-odirect", "disable O_DIRECT");

    po::positional_options_description p;
    p.add("file", 1);
    po::variables_map vm;
    try
    {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
        if (vm.count("help"))
        {
            cout << "gds_bench - GDS read + CUDA multi-stage parse benchmark\n\n";
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
    opt.delim = vm["delim"].as<char>();
    opt.odirect = (vm.count("no-odirect") == 0);
    return true;
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
    fs_block = static_cast<size_t>(st.st_blksize ? st.st_blksize : 4096);
    return true;
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


ssize_t single_big_gds_read(CUfileHandle_t cf_handle, char *d_buffer, size_t file_size)
{
    ssize_t total_read = 0;
    ssize_t ret = cuFileRead(cf_handle, d_buffer, file_size, 0, 0);
    if (ret < 0)
    {
        return ret;
    }
    total_read += ret;
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

void print_head(const void *buffer, size_t size)
{
    const char *data = static_cast<const char *>(buffer);
    size_t current_pos = 0;
    int newlines_found = 0;
    size_t print_limit = std::min(size, (size_t)1024);
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
     if (current_pos < size)
    {
        cout << "[...]\n";
    }
    cout << "---------------------------------\n";
}

void print_summary(size_t file_size, double read_seconds, const ParseResult& result, bool odirect)
{
    double file_mb = double(file_size) / (1024.0 * 1024.0);
    cout << fixed << setprecision(3);
    cout << "\n==== BENCHMARK SUMMARY ====" << "\n";
    cout << "O_DIRECT used: " << (odirect ? "YES" : "NO") << "\n";
    cout << "File size: " << file_size << " bytes (" << file_mb << " MB)" << "\n";
    cout << "\n-- READ STAGE --\n";
    cout << "Elapsed read time: " << read_seconds << " s\n";
    cout << "Read throughput: " << (file_mb / max(read_seconds, 1e-12)) << " MB/s\n";
    
    cout << "\n-- PARSE STAGE --\n";
    cout << "Parsed lines: " << result.total_lines << "\n";
    cout << "Parsed numeric items: " << result.total_items << "\n";
    cout << "Parse elapsed: " << result.parse_seconds << " s\n";
    cout << "Parse throughput (fileMB/parse_time): " << (file_mb / max(result.parse_seconds, 1e-12)) << " MB/s\n";

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    cout << "\nPeak RSS: " << ru.ru_maxrss << " KB\n";
    cout << "===========================\n";
}


ParseResult run_parse_stage(char *d_buffer, size_t file_size, char delim)
{
    auto t0_parse = clk::now();

    // Configure kernel launch parameters
    int blockSize = 256;
    int gridSize = static_cast<int>((file_size + blockSize - 1) / blockSize);

    // --- STAGE 1: Count new lines ---
    unsigned int *d_num_lines;
    CUDA_CHECK(cudaMalloc(&d_num_lines, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_num_lines, 0, sizeof(unsigned int)));
    num_new_lines_kernel<<<gridSize, blockSize>>>(d_buffer, file_size, d_num_lines);
    CUDA_CHECK(cudaGetLastError());
    
    unsigned int h_num_lines = 0;
    CUDA_CHECK(cudaMemcpy(&h_num_lines, d_num_lines, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    // Assuming file ends with a newline, otherwise logic to add 1 might be needed.
    size_t total_lines = h_num_lines;
    cout << "Parse Stage: Found " << total_lines << " lines.\n";

    if (total_lines == 0) {
        // Handle empty or single-line file
        CUDA_CHECK(cudaFree(d_num_lines));
        return {nullptr, 0, 0, 0.0};
    }

    // --- STAGE 2: Find newline indices ---
    unsigned long long *d_newline_indices;
    CUDA_CHECK(cudaMalloc(&d_newline_indices, (total_lines + 1) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_newline_indices, 0, (total_lines + 1) * sizeof(unsigned long long)));
    find_new_lines_kernel<<<gridSize, blockSize>>>(d_buffer, file_size, d_newline_indices);
    CUDA_CHECK(cudaGetLastError());
    // We need to sort the indices found atomically. A simple device-side sort will do.
    thrust::sort(thrust::device_pointer_cast(d_newline_indices + 1), thrust::device_pointer_cast(d_newline_indices + 1 + total_lines));


    // --- STAGE 3: Get items per line ---
    int *d_items_per_line;
    CUDA_CHECK(cudaMalloc(&d_items_per_line, total_lines * sizeof(int)));
    int itemsGridSize = (total_lines + blockSize - 1) / blockSize;
    get_items_per_line_kernel<<<itemsGridSize, blockSize>>>(d_buffer, d_newline_indices, total_lines, d_items_per_line, delim);
    CUDA_CHECK(cudaGetLastError());

    // --- STAGE 4: Prefix Sum to get offsets and total items ---
    unsigned long long *d_item_offsets;
    CUDA_CHECK(cudaMalloc(&d_item_offsets, (total_lines + 1) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_item_offsets, 0, (total_lines + 1) * sizeof(unsigned long long)));

    // Using Thrust for exclusive scan to get starting positions for each line's items
    thrust::exclusive_scan(thrust::device_pointer_cast(d_items_per_line),
                           thrust::device_pointer_cast(d_items_per_line + total_lines),
                           thrust::device_pointer_cast(d_item_offsets));
    
    // Get the total number of items from the last element of the offsets + last line's item count
    uint64_t total_items;
    int last_line_items;
    CUDA_CHECK(cudaMemcpy(&last_line_items, d_items_per_line + total_lines - 1, sizeof(int), cudaMemcpyDeviceToHost));
    unsigned long long last_offset;
    CUDA_CHECK(cudaMemcpy(&last_offset, d_item_offsets + total_lines-1, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    total_items = last_offset + last_line_items;

    cout << "Parse Stage: Found " << total_items << " total numeric items.\n";
    

    // --- STAGE 5: Allocate final buffer and convert ---
    unsigned int *d_final_data;
    CUDA_CHECK(cudaMalloc(&d_final_data, total_items * sizeof(unsigned int)));

    convert_char_to_int_kernel<<<itemsGridSize, blockSize>>>(d_buffer, d_newline_indices, d_item_offsets, total_lines, d_final_data, delim);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1_parse = clk::now();
    double parse_seconds = chrono::duration<double>(t1_parse - t0_parse).count();
    
    // Free intermediate buffers
    CUDA_CHECK(cudaFree(d_num_lines));
    CUDA_CHECK(cudaFree(d_newline_indices));
    CUDA_CHECK(cudaFree(d_items_per_line));
    CUDA_CHECK(cudaFree(d_item_offsets));

    cerr << "[timing] parse total: " << parse_seconds << " s\n";

    ParseResult result;
    result.d_final_data = d_final_data;
    result.total_lines = total_lines;
    result.total_items = total_items;
    result.parse_seconds = parse_seconds;

    return result;
}


// -----------------------------
// Main
// -----------------------------
int main(int argc, char **argv)
{
    Options opt;
    if (!parse_cli(argc, argv, opt))
        return 1;

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

    // GDS requires allocations to be aligned to FS block size for O_DIRECT
    size_t registration_size = round_up(file_size, fs_block);

    cout << "Starting GDS benchmark\n";
    cout << "File: " << opt.filename << " size: " << file_size << " bytes\n";
    cout << "Alignment / FS block: " << fs_block << " bytes\n";
    cout << "Alloc/register size (rounded): " << registration_size << " bytes\n";

    if (!init_gds_driver())
    {
        close(fd);
        return 1;
    }

    CUfileDescr_t cf_desc;
    memset(&cf_desc, 0, sizeof(CUfileDescr_t));
    cf_desc.handle.fd = fd;
    cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileHandle_t cf_handle;
    GDS_CHECK(cuFileHandleRegister(&cf_handle, &cf_desc));

    char *d_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buffer, registration_size));
    GDS_CHECK(cuFileBufRegister(d_buffer, registration_size, 0));

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
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1_transfer = clk::now();
    double transfer_seconds = chrono::duration<double>(t1_transfer - t0_transfer).count();

    // Run the full parsing pipeline
    ParseResult parse_result = run_parse_stage(d_buffer, file_size, opt.delim);

    size_t head_copy_size = min(file_size, (size_t)1024);
    vector<char> host_head(head_copy_size);
    CUDA_CHECK(cudaMemcpy(host_head.data(), d_buffer, head_copy_size, cudaMemcpyDeviceToHost));
    print_head(host_head.data(), head_copy_size);

    print_summary(file_size, transfer_seconds, parse_result, opt.odirect);

    // Final verification: Print last element from the parsed data
    if (parse_result.d_final_data && parse_result.total_items > 0) {
        unsigned int last_element;
        CUDA_CHECK(cudaMemcpy(&last_element, parse_result.d_final_data + parse_result.total_items - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        cout << "\nVerification: Last parsed integer is " << last_element << std::endl;
    }


    // Cleanup
    GDS_CHECK(cuFileBufDeregister(d_buffer));
    cuFileHandleDeregister(cf_handle);
    GDS_CHECK(cuFileDriverClose());
    close(fd);
    CUDA_CHECK(cudaFree(d_buffer));
    if (parse_result.d_final_data) {
        CUDA_CHECK(cudaFree(parse_result.d_final_data));
    }

    return 0;
}