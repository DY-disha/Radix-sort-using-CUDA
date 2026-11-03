#include "radix_sort.cuh"
#include "radix_config.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void check_cuda(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

struct Buffer {
    uint32_t* ptr = nullptr;
    explicit Buffer(size_t count) {
        if (count) check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(uint32_t)));
    }
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check_cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

struct Event {
    cudaEvent_t value = nullptr;
    Event() { check_cuda(cudaEventCreate(&value)); }
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

// Matches the exact input generator from the existing benchmark_cub_sort.cu.
std::vector<uint32_t> generate_input(size_t n, const std::string& pattern) {
    std::vector<uint32_t> input(n);
    std::mt19937 rng(2026);
    for (size_t i = 0; i < n; ++i) {
        if (pattern == "random") input[i] = rng();
        else if (pattern == "all_equal") input[i] = UINT32_MAX;
        else if (pattern == "few_distinct") input[i] = static_cast<uint32_t>(i % 7);
        else if (pattern == "sorted") input[i] = static_cast<uint32_t>(i);
        else if (pattern == "reversed") input[i] = static_cast<uint32_t>(n - i);
        else throw std::invalid_argument("Unknown pattern: " + pattern);
    }
    return input;
}

double median(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("No samples");
    std::sort(values.begin(), values.end());
    size_t m = values.size() / 2;
    return values.size() % 2 ? values[m] : (values[m - 1] + values[m]) / 2.0;
}

double benchmark_cuda_case(const std::vector<uint32_t>& input,
                           int warmup_runs, int measured_runs) {
    const size_t n = input.size();
    if (n == 0 || n > static_cast<size_t>(INT32_MAX) || measured_runs <= 0 || warmup_runs < 0)
        throw std::invalid_argument("Invalid benchmark size or run count");
    const size_t bytes = n * sizeof(uint32_t);

    Stream stream;
    Buffer original(n), first(n), second(n);
    RadixWorkspace workspace(n); // Allocation OUTSIDE timed sort.
    Event start, stop;

    check_cuda(cudaMemcpy(original.ptr, input.data(), bytes, cudaMemcpyHostToDevice));

    // Device-to-device restore is ordered on the same stream, before start event.
    auto restore = [&]() {
        check_cuda(cudaMemcpyAsync(first.ptr, original.ptr, bytes,
                                   cudaMemcpyDeviceToDevice, stream.value));
    };
    auto sort = [&]() {
        return radix_sort_device(first.ptr, second.ptr, n, workspace, stream.value);
    };

    for (int i = 0; i < warmup_runs; ++i) {
        restore();
        sort();
    }
    check_cuda(cudaStreamSynchronize(stream.value));

    std::vector<double> samples;
    samples.reserve(measured_runs);
    for (int i = 0; i < measured_runs; ++i) {
        restore();
        check_cuda(cudaEventRecord(start.value, stream.value));
        sort();  // Exactly four 8-bit radix passes for the integrated configuration.
        check_cuda(cudaEventRecord(stop.value, stream.value));
        check_cuda(cudaEventSynchronize(stop.value));
        float milliseconds = 0.0f;
        check_cuda(cudaEventElapsedTime(&milliseconds, start.value, stop.value));
        samples.push_back(milliseconds);
    }

    // Check full-sort output outside timing.
    // radix_sort_device returns the final device pointer (currently 'first' after 4 passes).
    std::vector<uint32_t> expected = input;
    std::sort(expected.begin(), expected.end());
    std::vector<uint32_t> actual(n);
    check_cuda(cudaMemcpy(actual.data(), first.ptr, bytes, cudaMemcpyDeviceToHost));
    if (actual != expected) {
        size_t i = 0;
        while (i < n && actual[i] == expected[i]) ++i;
        throw std::runtime_error("CUDA radix sort mismatch at index " + std::to_string(i));
    }
    return median(samples);
}

int main() {
    try {
        const int warmup_runs = 5;
        const int measured_runs = 20;
        const std::vector<size_t> sizes = {10000, 100000, 1000000};
        const std::vector<std::string> patterns = {
            "random", "all_equal", "few_distinct", "sorted", "reversed"
        };
        int device = 0;
        check_cuda(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device));
        std::cerr << "GPU: " << properties.name << '\n';
        std::cerr << "CUDA configuration: B=" << radix_config::B
                  << " Q=" << radix_config::Q
                  << " r=" << radix_config::RADIX_BITS << '\n';
        std::cerr << "Warm-up runs: " << warmup_runs << '\n';
        std::cerr << "Measured runs: " << measured_runs << '\n';
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "implementation,n,pattern,median_ms,status\n";
        size_t passed = 0;
        for (size_t n : sizes) {
            for (const auto& pattern : patterns) {
                try {
                    const auto input = generate_input(n, pattern);
                    const double value = benchmark_cuda_case(input, warmup_runs, measured_runs);
                    std::cout << "CUDA," << n << ',' << pattern << ',' << value << ",PASS\n";
                    ++passed;
                } catch (const std::exception& e) {
                    throw std::runtime_error("n=" + std::to_string(n) + " pattern=" + pattern + ": " + e.what());
                }
            }
        }
        std::cerr << "PASS: All " << passed << " CUDA full-sort cases matched CPU std::sort.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
