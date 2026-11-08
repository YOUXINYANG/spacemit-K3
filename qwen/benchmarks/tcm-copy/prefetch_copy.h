#pragma once
#include <cstdint>

void rvv_prefetch64(void *dst, const void *src, int64_t size);
void rvv_prefetch128(void *dst, const void *src, int64_t size);
