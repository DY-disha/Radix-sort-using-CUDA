// Standalone histogram benchmark. Each shift uses the ORIGINAL input order.
#include "block_histogram.cuh"
#include "block_histogram_warp.cuh"
#include <chrono>
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
        check(cudaMalloc(reinterpret_cast<void**>(&p), std::max(size_t(1), n) * sizeof(unsigned int)));
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

using Histogram = cudaError_t (*)(const unsigned int*, size_t, unsigned, unsigned,
                                  unsigned int*, cudaStream_t);
const Histogram versions[] = {histogram_device, histogram_warp_device};
constexpr int batch_runs = 100, measured_pairs = 20;
void validate(Histogram fn, const std::vector<unsigned int>& input,
              unsigned shift, unsigned r, Buffer& data, Buffer& hist, Stream& stream) {
    const auto expected = histogram_cpu(input, shift, r);
    // Poison output first to detect missing writes on every validation.
    if (!expected.empty())
        check(cudaMemsetAsync(hist.p, 0xa5, expected.size() * sizeof(unsigned int), stream.s));
    check(fn(data.p, input.size(), shift, r, hist.p, stream.s));
    check(cudaStreamSynchronize(stream.s));
    std::vector<unsigned int> actual(expected.size());
    if (!actual.empty())
        check(cudaMemcpy(actual.data(), hist.p, actual.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    if (actual != expected) throw std::runtime_error("Histogram mismatch: n="
        + std::to_string(input.size()) + " shift=" + std::to_string(shift)
        + " r=" + std::to_string(r));
}
void boundary_tests() {
    const size_t t = radix_config::TILE;
    const std::vector<size_t> sizes = {0,1,31,32,33,255,256,257,t-1,t,t+1,2*t+1};
    size_t checks = 0;
    Stream stream;
    for (size_t n : sizes) {
        Buffer data(n), hist(block_count(n) * 256);
        for (const std::string pattern : {"random", "all_equal", "few_distinct"}) {
            const auto input = input_for(n, pattern);
            if (n) check(cudaMemcpy(data.p, input.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));
            for (unsigned r : {1u,2u,4u,8u})
                for (unsigned shift = 0; shift < 32; shift += r)
                    for (auto fn : versions) {
                        validate(fn, input, shift, r, data, hist, stream);
                        ++checks;
                    }
        }
    }
    std::cerr << "PASS: " << checks << " boundary/reference checks.\n";
}
void startup_warmup() {
    const size_t n = 1000000;
    auto input = input_for(n, "random");
    Buffer data(n), hist(block_count(n) * radix_config::NUM_BINS);
    Stream stream;
    check(cudaMemcpy(data.p, input.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    do {
        for (int i = 0; i < 100; ++i)
            for (auto fn : versions)
                check(fn(data.p, n, 0, radix_config::RADIX_BITS, hist.p, stream.s));
        check(cudaStreamSynchronize(stream.s));
    } while (std::chrono::steady_clock::now() < until);
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return (values[values.size()/2-1] + values[values.size()/2]) / 2;
}
void run_case(size_t n, const std::string& pattern) {
    const auto input = input_for(n, pattern);
    Stream stream;
    Buffer data(n), hist(block_count(n) * radix_config::NUM_BINS);
    Event start, stop;
    check(cudaMemcpy(data.p, input.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));
    for (unsigned shift = 0; shift < 32; shift += radix_config::RADIX_BITS) {
        for (auto fn : versions)
            validate(fn, input, shift, radix_config::RADIX_BITS, data, hist, stream);
        // Identical warm-up for both variants, outside measurement.
        for (int i = 0; i < 5 * batch_runs; ++i)
            for (auto fn : versions)
                check(fn(data.p, n, shift, radix_config::RADIX_BITS, hist.p, stream.s));
        check(cudaStreamSynchronize(stream.s));
        std::vector<double> samples[2];
        for (int pair = 0; pair < measured_pairs; ++pair) {
            // Alternate AB then BA, reducing systematic ordering bias.
            for (int position = 0; position < 2; ++position) {
                const int v = (pair + position) % 2;
                check(cudaEventRecord(start.e, stream.s));
                for (int repeat = 0; repeat < batch_runs; ++repeat)
                    check(versions[v](data.p, n, shift, radix_config::RADIX_BITS, hist.p, stream.s));
                check(cudaEventRecord(stop.e, stream.s));
                check(cudaEventSynchronize(stop.e));
                float ms = 0;
                check(cudaEventElapsedTime(&ms, start.e, stop.e));
                samples[v].push_back(double(ms) / batch_runs);
            }
        }
        for (auto fn : versions)
            validate(fn, input, shift, radix_config::RADIX_BITS, data, hist, stream);
        const double baseline = median(samples[0]), warp = median(samples[1]);
        std::cout << n << ',' << pattern << ',' << shift << ',' << radix_config::B << ','
                  << radix_config::Q << ',' << radix_config::RADIX_BITS << ',' << batch_runs << ','
                  << baseline << ',' << warp << ',' << baseline / warp << ",PASS\n";
    }
}
}
int main() {
    try {
        int device = 0;
        check(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device));
        std::cerr << "GPU: " << prop.name << "\n"
                  << "Startup warmup: >=1 second, both variants.\n"
                  << "Per case: 500 warmup calls per variant; 20 measured pairs; 100 calls per batch.\n"
                  << "AB/BA alternating order; original input at every shift; allocation/copies/checks excluded.\n"
                  << "Reported time is batched elapsed time per call, including possible launch gaps.\n";
        boundary_tests();
        startup_warmup();
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "n,pattern,shift,B,Q,r,batch_runs,baseline_ms,warp_ms,speedup,status\n";
        for (size_t n : {size_t(10000), size_t(100000), size_t(1000000)})
            for (const std::string pattern : {"random", "all_equal", "few_distinct", "sorted", "reversed"})
                run_case(n, pattern);
        std::cerr << "PASS: both variants matched CPU reference for all 60 benchmark cases.\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
