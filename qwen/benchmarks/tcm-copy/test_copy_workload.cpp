#include "copy_workload.h"
#include <cstdio>
#include <cstring>
#include <initializer_list>

static void copy_bytes(void *dst, const void *src, int64_t size) {
    std::memcpy(dst, src, static_cast<std::size_t>(size));
}

int main() {
    const unsigned char source[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::size_t counts[]{1, 2, 3, 4, 8};
    for (std::size_t slots : {std::size_t(1), std::size_t(3)}) {
        for (std::size_t count : counts) {
            unsigned char dst[]{0xa5, 0, 0, 0, 0xa5};
            const auto *last = copy_repeated(copy_bytes, dst + 1, source, 3, slots, count);
            const auto *expected = source + ((count - 1) % slots) * 3;
            if (last != expected || std::memcmp(dst + 1, expected, 3) != 0 ||
                dst[0] != 0xa5 || dst[4] != 0xa5) {
                std::fprintf(stderr, "FAIL: slots=%zu iterations=%zu\n", slots, count);
                return 1;
            }
        }
    }
    unsigned char dst = 0xa5;
    if (copy_repeated(copy_bytes, &dst, source, 1, 3, 0) != nullptr || dst != 0xa5)
        return 1;
    if (copy_repeated(copy_bytes, &dst, source, 1, 0, 1) != nullptr || dst != 0xa5)
        return 1;
    std::puts("PASS: one slot, source rotation, wraparound, empty work, guards");
}
