#include "check_copy.h"
#include "copy_workload.h"
#include "prefetch_copy.h"
#include <spine_tcm.h>
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

// Exact declaration from the pinned rvv_kernels.h; avoids unrelated GGML headers.
namespace spacemit_kernels { namespace rvv {
void memcpy1d(void *dst, const void *src, int64_t size);
}}

static void libc_copy(void *dst, const void *src, int64_t n) {
    std::memcpy(dst, src, static_cast<std::size_t>(n));
}

static int run(unsigned char *tcm, std::size_t capacity,
               unsigned char *src, std::size_t source_capacity,
               bool bench, bool stream) {
    constexpr std::size_t iterations = 200000;
    const Copy methods[] = {spacemit_kernels::rvv::memcpy1d, libc_copy,
                            rvv_prefetch64, rvv_prefetch128};
    const char *names[] = {"rvv_memcpy1d", "libc_memcpy",
                           "rvv_prefetch64", "rvv_prefetch128"};
    constexpr int method_count = sizeof(methods) / sizeof(methods[0]);
    std::puts("method,bytes,src_mod128,dst_mod128,round,iterations,ns_per_copy,payload_GB_s,source_slots,source_working_set_bytes");
    for (std::size_t size : {std::size_t(16384), std::size_t(18432), std::size_t(20480)}) {
        const std::size_t slots = stream ? source_capacity / size : 1;
        if (slots == 0) return 1;
        for (std::size_t shift : {std::size_t(0), std::size_t(8)}) {
            const std::size_t offset = 128 + shift;
            if (offset > capacity || size > capacity - offset) return 1;
            unsigned char *dst = tcm + offset;
            for (int method = 0; method < method_count; ++method) {
                for (unsigned pattern = 0; pattern < 3; ++pattern) {
                    for (std::size_t slot : {std::size_t(0), slots / 2, slots - 1}) {
                        unsigned char *sample = src + slot * size;
                        for (std::size_t i = 0; i < size; ++i)
                            sample[i] = static_cast<unsigned char>((i * 37 + (i >> 8) + pattern * 71 + slot) & 255);
                        std::memset(tcm, 0xa5, capacity);
                        methods[method](dst, sample, static_cast<int64_t>(size));
                        if (!copy_is_correct(tcm, capacity, offset, sample, size, 0xa5)) {
                            std::fprintf(stderr, "FAIL: %s size=%zu shift=%zu pattern=%u slot=%zu\n",
                                         names[method], size, shift, pattern, slot);
                            return 1;
                        }
                    }
                }
            }
            std::fprintf(stderr, "PASS: bytes=%zu src_mod128=%zu dst_mod128=%zu source_slots=%zu working_set=%zu\n",
                         size, reinterpret_cast<uintptr_t>(src) % 128,
                         reinterpret_cast<uintptr_t>(dst) % 128, slots, slots * size);
            if (!bench) continue;
            for (int round = 0; round < 3; ++round) {
                // Alternate order between rounds to reduce order bias.
                for (int step = 0; step < method_count; ++step) {
                    const int method = round % 2 == 0 ? step : method_count - 1 - step;
                    Copy copy = methods[method];
                    std::memset(tcm, 0xa5, capacity);
                    copy_repeated(copy, dst, src, size, slots, stream ? slots : 64);
                    asm volatile("" ::: "memory");
                    const auto start = std::chrono::steady_clock::now();
                    const auto *last_source = copy_repeated(copy, dst, src, size, slots, iterations);
                    asm volatile("" ::: "memory");
                    const auto end = std::chrono::steady_clock::now();
                    if (!copy_is_correct(tcm, capacity, offset, last_source, size, 0xa5)) {
                        std::fprintf(stderr, "FAIL: timed copy or guards\n");
                        return 1;
                    }
                    const double ns = std::chrono::duration<double, std::nano>(end - start).count()
                                      / iterations;
                    std::printf("%s,%zu,%zu,%zu,%d,%zu,%.3f,%.3f,%zu,%zu\n", names[method], size,
                                reinterpret_cast<uintptr_t>(src) % 128,
                                reinterpret_cast<uintptr_t>(dst) % 128,
                                round + 1, iterations, ns, size / ns, slots, slots * size);
                }
            }
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    bool bench = false;
    bool stream = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) bench = true;
        else if (std::strcmp(argv[i], "--stream") == 0) stream = true;
        else {
            std::fprintf(stderr, "Usage: %s [--bench] [--stream]\n", argv[0]);
            return 2;
        }
    }
    // Match ime.cpp: enter the AI scheduling domain before setting affinity.
    const int initial_cpu = sched_getcpu();
    if (initial_cpu < 8 || initial_cpu > 15) {
        const int fd = open("/proc/set_ai_thread", O_WRONLY);
        if (fd < 0) {
            std::perror("Cannot open /proc/set_ai_thread; no TCM block requested");
            return 1;
        }
        const ssize_t written = write(fd, "0", 1);
        if (written != 1) {
            if (written < 0) std::perror("Cannot write /proc/set_ai_thread");
            else std::fprintf(stderr, "Short write to /proc/set_ai_thread\n");
            close(fd);
            return 1;
        }
        close(fd);
    }
    // This instance maps CPU 8 to real TCM block 0 (mask 0x300).
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(8, &affinity);
    if (sched_setaffinity(0, sizeof(affinity), &affinity) != 0) {
        std::perror("Cannot bind CPU 8; no TCM block requested");
        return 1;
    }
    std::fprintf(stderr, "cpu=%d tcm_id=0 mode=%s workload=%s\n", sched_getcpu(),
                 bench ? "benchmark" : "correctness only", stream ? "stream" : "hot");
    spine_tcm_mem_info_t info{};
    if (!spine_tcm_is_available() || spine_tcm_mem_info(&info) != 0 ||
        info.is_fake_tcm || info.blk_num == 0 || info.blk_size < 32768) {
        std::fprintf(stderr, "Real TCM unavailable or block too small\n");
        return 1;
    }
    std::fprintf(stderr, "block_size=%zu block_num=%zu fake=%d\n",
                 info.blk_size, info.blk_num, info.is_fake_tcm);
    void *source_memory = nullptr;
    const std::size_t source_capacity = stream ? 64 * 1024 * 1024 : 32768;
    if (posix_memalign(&source_memory, 128, source_capacity) != 0) return 1;
    auto *source = static_cast<unsigned char *>(source_memory);
    for (std::size_t i = 0; i < source_capacity; ++i)
        source[i] = static_cast<unsigned char>((i * 37 + (i >> 8) + (i >> 16)) & 255);
    auto *tcm = static_cast<unsigned char *>(spine_tcm_mem_get(0));
    if (!tcm) {
        std::fprintf(stderr, "Cannot get TCM block 0\n");
        std::free(source_memory);
        return 1;
    }
    const int result = run(tcm, info.blk_size, source, source_capacity, bench, stream);
    const int released = spine_tcm_mem_free(0);
    std::free(source_memory);
    std::fprintf(stderr, "tcm_free_result=%d test_result=%d\n", released, result);
    return result != 0 || released != 0 ? 1 : 0;
}
