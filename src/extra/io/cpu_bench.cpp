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
#include <atomic>
#include <algorithm>
#include <cassert>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/resource.h>

#include <liburing.h>
#include <omp.h>

#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace std;
using clk = chrono::high_resolution_clock;

static inline size_t round_up(size_t x, size_t a) {
    return ((x + a - 1) / a) * a;
}

struct Options {
    string filename;
    int user_threads = 0;
    int queue_depth = 64;
    int io_size_kb = 1024;
    int block_size = 0;
    char delim = ',';
    int max_retries = 3;
};

bool parse_cli(int argc, char** argv, Options &opt) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "show help")
        ("file", po::value<string>()->required(), "input file (positional)")
        ("threads,t", po::value<int>()->default_value(0), "number of parse threads (0=OpenMP default)")
        ("queue-depth,q", po::value<int>()->default_value(64), "io_uring queue depth")
        ("io-size-kb,i", po::value<int>()->default_value(1024), "IO size per request (KB)")
        ("block-size,b", po::value<int>()->default_value(0), "alignment block size in bytes (0 => filesystem default)")
        ("delim,d", po::value<char>()->default_value(','), "delimiter for parse stage")
        ("retries,r", po::value<int>()->default_value(3), "max retries per failed chunk")
    ;
    po::positional_options_description p; p.add("file", 1);
    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
        if (vm.count("help")) {
            cout << "io_uring_bench_fixed_retry - read + parse benchmark\n\n";
            cout << "Usage: " << argv[0] << " <file> [options]\n\n" << desc << "\n";
            return false;
        }
        po::notify(vm);
    } catch (const po::error &e) {
        cerr << "Error parsing options: " << e.what() << "\n\n" << desc << "\n";
        return false;
    }
    opt.filename = vm["file"].as<string>();
    opt.user_threads = vm["threads"].as<int>();
    opt.queue_depth = vm["queue-depth"].as<int>();
    opt.io_size_kb = vm["io-size-kb"].as<int>();
    opt.block_size = vm["block-size"].as<int>();
    opt.delim = vm["delim"].as<char>();
    opt.max_retries = vm["retries"].as<int>();
    if (opt.queue_depth < 1) opt.queue_depth = 1;
    if (opt.io_size_kb <= 0) opt.io_size_kb = 1;
    if (opt.max_retries < 0) opt.max_retries = 0;
    return true;
}

int open_file_odirect(const string &path, bool &out_odirect) {
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd >= 0) { out_odirect = true; return fd; }
    int err = errno;
    cerr << "Warning: open(..., O_DIRECT) failed: " << strerror(err) << ". Falling back to buffered read.\n";
    fd = open(path.c_str(), O_RDONLY);
    if (fd >= 0) out_odirect = false;
    return fd;
}

bool get_file_info(int fd, size_t &out_file_size, size_t &out_block_size) {
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return false; }
    out_file_size = static_cast<size_t>(st.st_size);
    out_block_size = (st.st_blksize > 0) ? static_cast<size_t>(st.st_blksize) : 4096;
    return true;
}

void* allocate_aligned_buffer(size_t alloc_size, size_t align) {
    void* buf = nullptr;
    if (posix_memalign(&buf, align, alloc_size) != 0 || buf == nullptr) {
        cerr << "posix_memalign failed\n";
        return nullptr;
    }
    // touch first page lightly to avoid huge lazy-fault noise
    volatile char *p = (volatile char*)buf;
    p[0] = p[0];
    return buf;
}

// ---------- parse routines ----------
uint64_t parse_range_count(const char* data, size_t start, size_t end, char delim) {
    const char* p = data + start;
    const char* const e = data + end;
    uint64_t cnt = 0;
    while (p < e) {
        while (p < e && !((*p >= '0' && *p <= '9') || *p == '-')) ++p;
        if (p >= e) break;
        const char* num_start = p;
        if (*num_start == '-') ++num_start;
        if (num_start < e && *num_start >= '0' && *num_start <= '9') {
            ++cnt;
            p = num_start;
            while (p < e && *p >= '0' && *p <= '9') ++p;
            while (p < e && *p != delim && *p != '\n' && *p != '\r') ++p;
            if (p < e && (*p == delim || *p == '\n' || *p == '\r')) ++p;
        } else ++p;
    }
    return cnt;
}

pair<uint64_t,double> run_parse_stage(const void* buffer, size_t file_size, int user_threads, char delim) {
    if (user_threads > 0) omp_set_num_threads(user_threads);
    int threads = omp_get_max_threads();
    vector<size_t> t_start(threads), t_end(threads);
    for (int t = 0; t < threads; ++t) {
        t_start[t] = (file_size * (size_t)t) / threads;
        t_end[t] = (file_size * (size_t)(t+1)) / threads;
    }
    const char* data = static_cast<const char*>(buffer);
    for (int t = 1; t < threads; ++t) {
        size_t s = t_start[t];
        if (s > 0) {
            while (s < file_size && data[s-1] != '\n') ++s;
        }
        t_start[t] = s;
        t_end[t-1] = s;
    }
    t_end[threads-1] = file_size;

    uint64_t total_numbers = 0;
    auto t0 = clk::now();
    #pragma omp parallel for reduction(+:total_numbers)
    for (int tid = 0; tid < threads; ++tid) {
        size_t s = t_start[tid];
        size_t e = t_end[tid];
        if (s < e) total_numbers += parse_range_count(data, s, e, delim);
    }
    auto t1 = clk::now();
    return { total_numbers, chrono::duration<double>(t1 - t0).count() };
}

// ---------- I/O with retries ----------
double submit_and_wait_reads_with_retries(int fd, struct io_uring &ring, void* buffer,
        size_t file_size, size_t io_chunk, int queue_depth,
        atomic<size_t>& error_count, int max_retries) {

    size_t chunks = (file_size + io_chunk - 1) / io_chunk;
    // ensure alloc_size = chunks * io_chunk is available in buffer (caller does)
    vector<int> retries(chunks, 0);
    vector<char> completed(chunks, 0);

    size_t submitted = 0;
    size_t completed_count = 0;

    auto t0 = clk::now();

    while (completed_count < chunks) {
        // Submit up to queue_depth outstanding (counting submitted - completed_count)
        while (submitted < chunks && (submitted - completed_count) < static_cast<size_t>(queue_depth)) {
            // find next not submitted (submitted always grows monotonically here)
            size_t idx = submitted;
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            if (!sqe) break;
            size_t offset = idx * io_chunk;
            // always use full io_chunk length (aligned)
            size_t len = io_chunk;
            io_uring_prep_read(sqe, fd, static_cast<char*>(buffer) + offset, len, offset);
            io_uring_sqe_set_data(sqe, (void*)(uintptr_t)idx);
            submitted++;
        }

        // Submit queued SQEs
        int sret;
        do {
            sret = io_uring_submit(&ring);
            if (sret < 0 && sret != -EAGAIN) {
                cerr << "io_uring_submit failed: " << strerror(-sret) << "\n";
                // consider remaining pending as errors
                size_t remaining = submitted - completed_count;
                error_count += remaining;
                auto tnow = clk::now();
                return chrono::duration<double>(tnow - t0).count();
            }
        } while (sret < 0 && sret == -EAGAIN);

        // Wait for at least one completion, then drain available completions
        struct io_uring_cqe* cqe = nullptr;
        int wait_rc = io_uring_wait_cqe(&ring, &cqe);
        if (wait_rc < 0) {
            cerr << "io_uring_wait_cqe failed: " << strerror(-wait_rc) << "\n";
            error_count += (submitted - completed_count);
            auto tnow = clk::now();
            return chrono::duration<double>(tnow - t0).count();
        }

        // Process cqe and any available
        while (cqe) {
            uintptr_t idx = (uintptr_t)io_uring_cqe_get_data(cqe);
            int res = cqe->res;
            if (res < 0) {
                int err = -res;
                // decide retry vs permanent
                bool should_retry = false;
                if (err == EAGAIN || err == EINTR) should_retry = true;
                // Some drivers may return EIO for transient problems; be conservative and retry a couple times?
                if (err == EIO && retries[idx] < max_retries) should_retry = true;

                if (should_retry && retries[idx] < max_retries) {
                    retries[idx]++;
                    // resubmit this chunk immediately (try to keep queue full)
                    struct io_uring_sqe* nsqe = io_uring_get_sqe(&ring);
                    if (nsqe) {
                        size_t offset = idx * io_chunk;
                        size_t len = io_chunk;
                        io_uring_prep_read(nsqe, fd, static_cast<char*>(buffer) + offset, len, offset);
                        io_uring_sqe_set_data(nsqe, (void*)(uintptr_t)idx);
                        // We don't increase submitted here; this is a retry of already counted chunk.
                        int rc = io_uring_submit(&ring);
                        if (rc < 0) {
                            cerr << "Retry submit failed for chunk " << idx << ": " << strerror(-rc) << "\n";
                            error_count.fetch_add(1, std::memory_order_relaxed);
                            completed[idx] = 1;
                            ++completed_count;
                        } else {
                            // We'll wait for its completion later (do not mark completed yet)
                        }
                    } else {
                        // couldn't get sqe to retry; mark as error (conservative) and continue
                        cerr << "Could not allocate SQE to retry chunk " << idx << ", marking failed\n";
                        error_count.fetch_add(1, std::memory_order_relaxed);
                        completed[idx] = 1;
                        ++completed_count;
                    }
                } else {
                    // Permanent failure for this chunk
                    cerr << "AIO error (chunk " << idx << "): " << strerror(err) << " (retries=" << retries[idx] << ")\n";
                    error_count.fetch_add(1, std::memory_order_relaxed);
                    if (!completed[idx]) {
                        completed[idx] = 1;
                        ++completed_count;
                    }
                }
            } else {
                // Positive result: bytes read (may be < io_chunk for last chunk beyond file end)
                if (!completed[idx]) {
                    completed[idx] = 1;
                    ++completed_count;
                }
            }

            io_uring_cqe_seen(&ring, cqe);
            // peek for another
            struct io_uring_cqe* next = nullptr;
            int peek = io_uring_peek_cqe(&ring, &next);
            if (peek == 0 && next != nullptr) cqe = next;
            else cqe = nullptr;
        }
    }

    auto t1 = clk::now();
    return chrono::duration<double>(t1 - t0).count();
}

// ---------- printing & utilities ----------
void print_head(const void* buffer, size_t size) {
    const char* data = static_cast<const char*>(buffer);
    size_t pos = 0;
    int nfound = 0;
    size_t limit = std::min(size, (size_t)1024);
    cout << "\n--- FILE HEAD (first 2 lines) ---\n";
    while (pos < limit && nfound < 2) {
        cout.put(data[pos]);
        if (data[pos] == '\n') ++nfound;
        ++pos;
    }
    if (pos < size && nfound < 2) cout << "[...]\n";
    else if (pos < size) {
        if (pos > 0 && data[pos-1] != '\n') cout << "\n";
        cout << "[...]\n";
    }
    cout << "[...]\n";
    // cout << "---------------------------------\n";
}

void print_summary(size_t file_size, double read_seconds, uint64_t parsed_numbers, double parse_seconds, bool odirect, size_t io_errors) {
    double file_mb = double(file_size) / (1024.0 * 1024.0);
    cout << fixed << setprecision(3);
    cout << "\n==== BENCHMARK SUMMARY ====\n";
    cout << "O_DIRECT used: " << (odirect ? "YES" : "NO") << "\n";
    cout << "File size: " << file_size << " bytes (" << file_mb << " MB)\n";
    cout << "\n-- READ STAGE --\n";
    cout << "Failed I/O requests: " << io_errors << "\n";
    cout << "Elapsed read time: " << read_seconds << " s\n";
    if (io_errors == 0) cout << "Read throughput: " << (file_mb / max(read_seconds, 1e-12)) << " MB/s\n";
    else cout << "Read throughput: N/A (due to I/O errors)\n";
    cout << "\n-- PARSE STAGE --\n";
    cout << "Parsed numeric tokens: " << parsed_numbers << "\n";
    cout << "Parse elapsed: " << parse_seconds << " s\n";
    if (io_errors == 0) cout << "Parse throughput (fileMB/parse_time): " << (file_mb / max(parse_seconds, 1e-12)) << " MB/s\n";
    else cout << "Parse throughput: N/A (I/O errors)\n";
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    cout << "\nPeak RSS: " << ru.ru_maxrss << " KB\n";
    cout << "===========================\n";
}

// ---------- main ----------
int main(int argc, char** argv) {
    Options opt;
    if (!parse_cli(argc, argv, opt)) return 1;

    bool used_odirect = false;
    int fd = open_file_odirect(opt.filename, used_odirect);
    if (fd < 0) { perror("open"); return 1; }

    size_t file_size = 0, fs_block = 0;
    if (!get_file_info(fd, file_size, fs_block)) { close(fd); return 1; }
    if (file_size == 0) { cerr << "Empty file.\n"; close(fd); return 0; }

    size_t block_size = (opt.block_size > 0) ? static_cast<size_t>(opt.block_size) : fs_block;
    if (block_size == 0) block_size = 4096;

    // Ensure io_chunk is a multiple of block_size (very important for O_DIRECT)
    size_t requested_chunk = static_cast<size_t>(opt.io_size_kb) * 1024;
    size_t io_chunk = round_up(requested_chunk, block_size);

    // chunks (and alloc_size) use io_chunk as unit
    size_t chunks = (file_size + io_chunk - 1) / io_chunk;
    size_t alloc_size = chunks * io_chunk; // ensure we can read full aligned chunks
    cout << "Starting benchmark\n";
    cout << "File: " << opt.filename << " size: " << file_size << " bytes\n";
    cout << "block_size (alignment): " << block_size << "  io_chunk: " << io_chunk
         << "  queue_depth: " << opt.queue_depth << "  chunks: " << chunks << "\n";
    cout << "Parser delimiter: '" << opt.delim << "'\n";

    void* bigbuf = allocate_aligned_buffer(alloc_size, block_size);
    if (!bigbuf) { close(fd); return 1; }

    struct io_uring ring;
    if (io_uring_queue_init(static_cast<unsigned>(opt.queue_depth), &ring, 0) != 0) {
        cerr << "io_uring_queue_init failed\n";
        free(bigbuf); close(fd); return 1;
    }

    atomic<size_t> io_errors{0};
    double read_seconds = submit_and_wait_reads_with_retries(fd, ring, bigbuf, file_size, io_chunk, opt.queue_depth, io_errors, opt.max_retries);

    io_uring_queue_exit(&ring);

    if (io_errors.load() > 0) {
        cerr << "\nWARNING: Benchmark completed, but " << io_errors.load() << " I/O errors occurred during the read stage.\n";
        cerr << "The resulting buffer may be incomplete or contain garbage data.\n";
    }

    // parse only up to file_size (not alloc_size)
    auto parse_res = run_parse_stage(bigbuf, file_size, opt.user_threads, opt.delim);

    // print first lines (safe: copy limited)
    size_t head_copy = min(file_size, (size_t)1024);
    print_head(bigbuf, head_copy);
    print_summary(file_size, read_seconds, parse_res.first, parse_res.second, used_odirect, io_errors.load());

    free(bigbuf);
    close(fd);
    return io_errors.load() > 0 ? 1 : 0;
}
