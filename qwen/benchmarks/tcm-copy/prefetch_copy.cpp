#include "prefetch_copy.h"
#include <cstddef>

namespace spacemit_kernels { namespace rvv {
void memcpy1d(void *dst, const void *src, int64_t size);
}}

// Experimental prefetch sweep before the unchanged, exported RVV copy.
// The interval is a test parameter, not an assumed hardware cache-line size.
template <std::size_t Step>
static void prefetch_then_copy(void *dst, const void *src, int64_t size) {
    if (size <= 0) return;
    const auto *bytes = static_cast<const unsigned char *>(src);
    for (std::size_t i = 0; i < static_cast<std::size_t>(size); i += Step)
        __builtin_prefetch(bytes + i, 0, 3);
    spacemit_kernels::rvv::memcpy1d(dst, src, size);
}

void rvv_prefetch64(void *dst, const void *src, int64_t size) {
    prefetch_then_copy<64>(dst, src, size);
}

void rvv_prefetch128(void *dst, const void *src, int64_t size) {
    prefetch_then_copy<128>(dst, src, size);
}
