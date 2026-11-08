#pragma once
#include <cstddef>
#include <cstring>

inline bool copy_is_correct(const unsigned char *region, std::size_t capacity,
                            std::size_t offset, const unsigned char *src,
                            std::size_t size, unsigned char sentinel) {
    if (offset > capacity || size > capacity - offset) return false;
    if (std::memcmp(region + offset, src, size) != 0) return false;
    for (std::size_t i = 0; i < offset; ++i)
        if (region[i] != sentinel) return false;
    for (std::size_t i = offset + size; i < capacity; ++i)
        if (region[i] != sentinel) return false;
    return true;
}
