#include "global_offsets/global_offsets.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

void check_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

double benchmark_case(
    size_t num_blocks,
    size_t num_bins,
    int warmup_runs,
    int measured_runs
) {
    const size_t n = num_blocks * num_bins;
    const size_t bytes = n * sizeof(uint32_t);

    std::vector<uint32_t> hist(n);

    // Deterministic synthetic histogram.
    for (size_t block = 0; block < num_blocks; ++block) {
        for (size_t bin = 0; bin < num_bins; ++bin) {
            hist[block * num_bins + bin] =
                static_cast<uint32_t>(
                    (block * 7 + bin * 3 + 1) % 11
                );
        }
    }

    uint32_t* d_hist = nullptr;
    uint32_t* d_hist_T = nullptr;
    uint32_t* d_offsets = nullptr;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    try {
        check_cuda(cudaMalloc(
            reinterpret_cast<void**>(&d_hist),
            bytes
        ));

        check_cuda(cudaMalloc(
            reinterpret_cast<void**>(&d_hist_T),
            bytes
        ));

        check_cuda(cudaMalloc(
            reinterpret_cast<void**>(&d_offsets),
            bytes
        ));

        check_cuda(cudaMemcpy(
            d_hist,
            hist.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        // Warm-up runs.
        for (int i = 0; i < warmup_runs; ++i) {
            check_cuda(
                compute_global_offsets(
                    d_hist,
                    d_hist_T,
                    d_offsets,
                    num_blocks,
                    num_bins
                )
            );
        }

        check_cuda(cudaDeviceSynchronize());

        check_cuda(cudaEventCreate(&start));
        check_cuda(cudaEventCreate(&stop));

        check_cuda(cudaEventRecord(start));

        for (int i = 0; i < measured_runs; ++i) {
            check_cuda(
                compute_global_offsets(
                    d_hist,
                    d_hist_T,
                    d_offsets,
                    num_blocks,
                    num_bins
                )
            );
        }

        check_cuda(cudaEventRecord(stop));
        check_cuda(cudaEventSynchronize(stop));

        float total_ms = 0.0f;

        check_cuda(cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        ));

        check_cuda(cudaEventDestroy(start));
        start = nullptr;

        check_cuda(cudaEventDestroy(stop));
        stop = nullptr;

        check_cuda(cudaFree(d_hist));
        d_hist = nullptr;

        check_cuda(cudaFree(d_hist_T));
        d_hist_T = nullptr;

        check_cuda(cudaFree(d_offsets));
        d_offsets = nullptr;

        return static_cast<double>(total_ms)
             / static_cast<double>(measured_runs);

    } catch (...) {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }

        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }

        if (d_hist != nullptr) {
            cudaFree(d_hist);
        }

        if (d_hist_T != nullptr) {
            cudaFree(d_hist_T);
        }

        if (d_offsets != nullptr) {
            cudaFree(d_offsets);
        }

        throw;
    }
}

int main() {
    try {
        const int warmup_runs = 20;
        const int measured_runs = 100;

        const std::vector<std::pair<size_t, size_t>> shapes = {
            {32, 16},       // 512
            {64, 16},       // 1024
            {256, 16},      // 4096
            {625, 16},      // 10000
            {4096, 16},     // 65536
            {16384, 16},    // 262144
            {65536, 16}     // 1048576
        };

        std::cout
            << "blocks,bins,entries,avg_ms\n";

        for (const auto& shape : shapes) {
            const size_t blocks = shape.first;
            const size_t bins = shape.second;
            const size_t n = blocks * bins;

            const double avg_ms =
                benchmark_case(
                    blocks,
                    bins,
                    warmup_runs,
                    measured_runs
                );

            std::cout
                << blocks << ","
                << bins << ","
                << n << ","
                << avg_ms << "\n";
        }

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "ERROR: "
            << error.what()
            << "\n";

        return 1;
    }
}