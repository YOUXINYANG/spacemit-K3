#include "check_copy.h"
#include <cstdio>
#include <vector>

int main() {
    const std::vector<unsigned char> src{1, 2, 3, 4};
    std::vector<unsigned char> region(16, 0xa5);
    std::memcpy(region.data() + 5, src.data(), src.size());
    auto valid = [&] {
        return copy_is_correct(region.data(), region.size(), 5,
                               src.data(), src.size(), 0xa5);
    };
    if (!valid()) return 1;
    region[6] ^= 1;
    if (valid()) { std::puts("FAIL: missed payload corruption"); return 1; }
    region[6] ^= 1;
    region[4] ^= 1;
    if (valid()) { std::puts("FAIL: missed write before destination"); return 1; }
    region[4] ^= 1;
    region[9] ^= 1;
    if (valid()) { std::puts("FAIL: missed write after destination"); return 1; }
    if (copy_is_correct(region.data(), region.size(), 15,
                        src.data(), src.size(), 0xa5)) return 1;
    std::puts("PASS: payload, both guards, bounds");
}
