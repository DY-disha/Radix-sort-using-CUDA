// Standalone histogram benchmark. Each shift uses the ORIGINAL input order.
#include "block_histogram.cuh"
#include "radix_config.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
    unsigned int* p = nullptr;
    explicit Buffer(size_t n) {
        check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(unsigned int)));
    }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Stream {
    cudaStream_t s = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { if (s) cudaStreamDestroy(s); }
};
struct Event {
    cudaEvent_t e = nullptr;
    Event() { check(cudaEventCreate(&e)); }
    ~Event() { if (e) cudaEventDestroy(e); }
};
std::vector<unsigned int> input_for(size_t n, const std::string& pattern) {
    std::vector<unsigned int> a(n);
    std::mt19937 rng(2026);
    for (size_t i = 0; i < n; ++i) {
        if (pattern == "random") a[i] = rng();
        else if (pattern == "all_equal") a[i] = UINT32_MAX;
        else if (pattern == "few_distinct") a[i] = unsigned(i % 7);
        else if (pattern == "sorted") a[i] = unsigned(i);
        else if (pattern == "reversed") a[i] = unsigned(n - i);
        else throw std::invalid_argument("Unknown pattern");
    }
    return a;
}
void run_case(size_t n, const std::string& pattern) {
    constexpr int warmups = 5, runs = 20;
    const auto input = input_for(n, pattern);
    const size_t entries = block_count(n) * radix_config::NUM_BINS;
    Stream stream;
    Buffer data(n), hist(entries);
    Event start, stop;
    check(cudaMemcpy(data.p, input.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));
    for (unsigned shift = 0; shift < 32; shift += radix_config::RADIX_BITS) {
        auto launch = [&]() {
            check(histogram_device(data.p, n, shift, radix_config::RADIX_BITS, hist.p, stream.s));
        };
        for (int i = 0; i < warmups; ++i) launch();
        check(cudaStreamSynchronize(stream.s));
        std::vector<double> samples;
        for (int i = 0; i < runs; ++i) {
            constexpr int batch_runs = 100;

            check(cudaEventRecord(start.e, stream.s));

            for (int repeat = 0; repeat < batch_runs; ++repeat) {
                launch();
            }

            check(cudaEventRecord(stop.e, stream.s));
            check(cudaEventSynchronize(stop.e));
            float ms = 0;
            check(cudaEventElapsedTime(&ms, start.e, stop.e));
            samples.push_back(static_cast<double>(ms) / batch_runs);
        }
        // CPU reference and device-to-host copy are outside timing.
        const auto expected = histogram_cpu(input, shift, radix_config::RADIX_BITS);
        std::vector<unsigned int> actual(entries);
        check(cudaMemcpy(actual.data(), hist.p, entries * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if (actual != expected) throw std::runtime_error("Histogram mismatch: n=" + std::to_string(n)
            + " pattern=" + pattern + " shift=" + std::to_string(shift));
        std::sort(samples.begin(), samples.end());
        const double median = (samples[9] + samples[10]) / 2.0;
        std::cout << n << ',' << pattern << ',' << shift << ',' << radix_config::B << ','
                  << radix_config::Q << ',' << radix_config::RADIX_BITS << ','
                  << radix_config::NUM_BINS << ',' << block_count(n) << ','
                  << median << ',' << samples.front() << ',' << samples.back() << ",PASS\n";
    }
}
}
int main() {
    try {
        int device = 0;
        check(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device));
        std::cerr << "GPU: " << prop.name << "\nWarmups: 5; measurements: 20\n"
                  << "Standalone histogram; original input order for every shift.\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "n,pattern,shift,B,Q,r,bins,blocks,median_ms,min_ms,max_ms,status\n";
        for (size_t n : {size_t(10000), size_t(100000), size_t(1000000)})
            for (const std::string pattern : {"random", "all_equal", "few_distinct", "sorted", "reversed"})
                run_case(n, pattern);
        std::cerr << "PASS: all 60 histogram cases matched CPU reference.\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
