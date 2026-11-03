// Complete unsigned 32-bit radix-sort example. Tests have separate entry points.
#include "radix_sort.cuh"
#include "radix_config.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct DeviceBuffer {
    uint32_t* data = nullptr;
    explicit DeviceBuffer(size_t n) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(uint32_t)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: ./build/radix_sort [unsigned integers...]\n"
                      << "No arguments: sort a built-in example.\n";
            return 0;
        }
        std::vector<uint32_t> values;
        for (int i = 1; i < argc; ++i) {
            const std::string token(argv[i]);
            if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("Expected an unsigned decimal integer: " + token);
            const unsigned long long number = std::stoull(token);
            if (number > std::numeric_limits<uint32_t>::max())
                throw std::out_of_range("Value exceeds uint32 range: " + token);
            values.push_back(static_cast<uint32_t>(number));
        }
        if (argc == 1) values = {7, 0, 4294967295u, 3, 7, 2147483648u};
        auto expected = values;
        std::sort(expected.begin(), expected.end());
        DeviceBuffer input(values.size()), scratch(values.size());
        RadixWorkspace workspace(values.size());
        const size_t bytes = values.size() * sizeof(uint32_t);
        check(cudaMemcpy(input.data, values.data(), bytes, cudaMemcpyHostToDevice));
        uint32_t* sorted = radix_sort_device(input.data, scratch.data, values.size(), workspace);
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(values.data(), sorted, bytes, cudaMemcpyDeviceToHost));
        if (values != expected) throw std::runtime_error("Result differs from CPU std::sort");
        std::cout << "Sorted:";
        for (uint32_t x : values) std::cout << ' ' << x;
        std::cout << "\nPASS: complete radix sort matches CPU std::sort.\n"
                  << "B=" << radix_config::B << " Q=" << radix_config::Q
                  << " r=" << radix_config::RADIX_BITS << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
