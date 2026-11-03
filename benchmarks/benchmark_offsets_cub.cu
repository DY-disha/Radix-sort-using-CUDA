#include "global_offsets/transpose.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

void check_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

double benchmark_cub_case(
    size_t num_blocks,
    size_t num_bins,
    int warmup_runs,
    int measured_runs
) {
    if (num_bins == 0) {
        throw std::invalid_argument("num_bins must be positive");
    }

    if (num_blocks >
        std::numeric_limits<size_t>::max() / num_bins) {

        throw std::overflow_error(
            "Histogram dimensions overflow size_t"
        );
    }

    const size_t n = num_blocks * num_bins;

    if (n == 0) {
        return 0.0;
    }

    const size_t bytes =
        n * sizeof(uint32_t);

    std::vector<uint32_t> hist(n);

    // Same deterministic histogram used in our benchmark.
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

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    try {
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_hist),
                bytes
            )
        );

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_hist_T),
                bytes
            )
        );

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_offsets),
                bytes
            )
        );

        check_cuda(
            cudaMemcpy(
                d_hist,
                hist.data(),
                bytes,
                cudaMemcpyHostToDevice
            )
        );

        // ----------------------------------------------------
        // Query CUB temporary-storage requirement.
        // ----------------------------------------------------
        check_cuda(
            cub::DeviceScan::ExclusiveSum(
                nullptr,
                temp_storage_bytes,
                d_hist_T,
                d_offsets,
                n
            )
        );

        check_cuda(
            cudaMalloc(
                &d_temp_storage,
                temp_storage_bytes
            )
        );

        // ----------------------------------------------------
        // Warm-up
        // ----------------------------------------------------
        for (int i = 0; i < warmup_runs; ++i) {

            transpose_histogram<<<1, 256>>>(
                d_hist,
                d_hist_T,
                num_blocks,
                num_bins
            );

            check_cuda(cudaGetLastError());

            check_cuda(
                cub::DeviceScan::ExclusiveSum(
                    d_temp_storage,
                    temp_storage_bytes,
                    d_hist_T,
                    d_offsets,
                    n
                )
            );
        }

        check_cuda(cudaDeviceSynchronize());

        check_cuda(cudaEventCreate(&start));
        check_cuda(cudaEventCreate(&stop));

        // ----------------------------------------------------
        // Timed region
        // ----------------------------------------------------
        check_cuda(cudaEventRecord(start));

        for (int i = 0; i < measured_runs; ++i) {

            transpose_histogram<<<1, 256>>>(
                d_hist,
                d_hist_T,
                num_blocks,
                num_bins
            );

            check_cuda(cudaGetLastError());

            check_cuda(
                cub::DeviceScan::ExclusiveSum(
                    d_temp_storage,
                    temp_storage_bytes,
                    d_hist_T,
                    d_offsets,
                    n
                )
            );
        }

        check_cuda(cudaEventRecord(stop));
        check_cuda(cudaEventSynchronize(stop));

        float total_ms = 0.0f;

        check_cuda(
            cudaEventElapsedTime(
                &total_ms,
                start,
                stop
            )
        );

        check_cuda(cudaEventDestroy(start));
        start = nullptr;

        check_cuda(cudaEventDestroy(stop));
        stop = nullptr;

        check_cuda(cudaFree(d_temp_storage));
        d_temp_storage = nullptr;

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

        if (d_temp_storage != nullptr) {
            cudaFree(d_temp_storage);
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

        const std::vector<
            std::pair<size_t, size_t>
        > shapes = {
            {32, 16},       // 512
            {64, 16},       // 1024
            {256, 16},      // 4096
            {625, 16},      // 10000
            {4096, 16},     // 65536
            {16384, 16},    // 262144
            {65536, 16}     // 1048576
        };

        std::cout
            << "blocks,bins,entries,cub_avg_ms\n";

        for (const auto& shape : shapes) {
            const size_t blocks = shape.first;
            const size_t bins = shape.second;
            const size_t n = blocks * bins;

            const double avg_ms =
                benchmark_cub_case(
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