#include <array>
#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>

using Hist = std::array<std::uint64_t, 256>;

Hist histogram(const std::uint8_t* data, std::size_t n) {
    Hist h{};
    for (std::size_t i = 0; i < n; ++i) ++h[data[i]];
    return h;
}

int main() {
    const std::vector<std::uint8_t> sample =
        {1,2,3,1,2,4,5,1,3,2,7,1};
    const auto h = histogram(sample.data(), sample.size());
    std::cout << "CPU UINT8 HISTOGRAM\n";
    for (int i = 0; i < 256; ++i)
        if (h[i]) std::cout << "Bin " << i << ": " << h[i] << '\n';
    std::uint64_t total = 0;
    for (auto x : h) total += x;
    std::cout << "Total: " << total << " (expected " << sample.size() << ")\n";
    return total == sample.size() ? 0 : 1;
}
