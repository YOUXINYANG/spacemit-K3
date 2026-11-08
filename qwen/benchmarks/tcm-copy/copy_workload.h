#pragma once
#include <cstddef>
#include <cstdint>

using Copy = void (*)(void *, const void *, int64_t);

inline const unsigned char *copy_repeated(Copy copy, void *dst,
                                         const unsigned char *src,
                                         std::size_t size, std::size_t slots,
                                         std::size_t iterations) {
    if (size == 0 || slots == 0 || iterations == 0) return nullptr;
    const unsigned char *cursor = src;
    const unsigned char *last = nullptr;
    const unsigned char *end = src + size * slots;
    for (std::size_t i = 0; i < iterations; ++i) {
        copy(dst, cursor, static_cast<int64_t>(size));
        last = cursor;
        cursor += size;
        if (cursor == end) cursor = src;
    }
    return last;
}
