#include "radix_sort.cuh"
#include "radix_config.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
    uint32_t* p = nullptr;
    explicit Buffer(size_t n) {
        if (n) check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(uint32_t)));
    }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Stream {
    cudaStream_t s;
    Stream() { check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(s); }
};
void compare(const uint32_t* gpu, const std::vector<uint32_t>& expected) {
    std::vector<uint32_t> actual(expected.size());
    if (!actual.empty()) check(cudaMemcpy(actual.data(), gpu,
        actual.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (actual != expected) {
        size_t i = 0;
        while (i < actual.size() && actual[i] == expected[i]) ++i;
        throw std::runtime_error("Mismatch at index " + std::to_string(i));
    }
}
void restore(Buffer& a, const std::vector<uint32_t>& input, cudaStream_t stream) {
    if (!input.empty()) check(cudaMemcpyAsync(a.p, input.data(),
        input.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
}
void run_case(const std::vector<uint32_t>& input, cudaStream_t stream) {
    const size_t n = input.size();
    Buffer a(n), b(n);
    RadixWorkspace w(n);
    // Each digit pass is compared to an independent stable comparison sort.
    // Different full keys sharing a digit expose order errors within that bin.
    for (unsigned shift : {0u, 8u, 16u, 24u}) {
        auto expected = input;
        std::stable_sort(expected.begin(), expected.end(), [shift](uint32_t x, uint32_t y) {
            return ((x >> shift) & 255u) < ((y >> shift) & 255u);
        });
        restore(a, input, stream);
        radix_pass_device(a.p, b.p, n, shift, w, stream);
        check(cudaStreamSynchronize(stream));
        compare(b.p, expected);
    }
    auto expected = input;
    std::sort(expected.begin(), expected.end());
    // Reuse the same workspace twice to catch stale state between sorts.
    for (int repeat = 0; repeat < 2; ++repeat) {
        restore(a, input, stream);
        auto result = radix_sort_device(a.p, b.p, n, w, stream);
        check(cudaStreamSynchronize(stream));
        compare(result, expected);
    }
}
int main(int argc, char** argv) {
    try {
        Stream stream;
        const size_t T = radix_config::TILE;
        std::vector<size_t> sizes = {0, 1, 31, 32, 33, 255, 256, 257,
                                   T-1, T, T+1, 2*T+1, 33*T+17};
        // Optional large case crosses another level of the hierarchical scan.
        if (argc > 1 && std::string(argv[1]) == "--large") sizes.push_back(1024*T+1);
        std::mt19937 rng(2026);
        size_t cases = 0;
        for (size_t n : sizes) {
            for (unsigned pattern = 0; pattern < 6; ++pattern) {
                std::vector<uint32_t> input(n);
                for (size_t i = 0; i < n; ++i) {
                    switch (pattern) {
                        case 0: input[i] = rng(); break;
                        case 1: input[i] = UINT32_MAX; break;
                        case 2: input[i] = uint32_t(i); break;
                        case 3: input[i] = uint32_t(n-i); break;
                        case 4: input[i] = uint32_t(i%7); break;
                        // Same low byte, different upper bytes: stability is observable.
                        default: input[i] = (uint32_t(n-i) << 8) | 7u; break;
                    }
                }
                try { run_case(input, stream.s); }
                catch (const std::exception& e) {
                    throw std::runtime_error("N=" + std::to_string(n) +
                        " pattern=" + std::to_string(pattern) + ": " + e.what());
                }
                ++cases;
            }
        }
        std::cout << "RADIX SORT VALID: " << cases << " datasets; " << cases*4
                  << " stable digit-pass checks; " << cases*2 << " full-sort checks.\n"
                  << "B=" << radix_config::B << " Q=" << radix_config::Q
                  << " r=" << radix_config::RADIX_BITS << " TILE=" << T << '\n';
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
