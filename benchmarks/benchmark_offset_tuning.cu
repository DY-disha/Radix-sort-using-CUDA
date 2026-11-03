
#include "global_offsets/global_offsets.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

void check_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

// Automatically free GPU memory.
struct DeviceBuffer {
    uint32_t* ptr = nullptr;

    explicit DeviceBuffer(size_t elements) {
        if (elements > 0) {
            check_cuda(cudaMalloc(
                reinterpret_cast<void**>(&ptr),
                elements * sizeof(uint32_t)
            ));
        }
    }

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Automatically destroy CUDA events.
struct EventPair {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    EventPair() {
        check_cuda(cudaEventCreate(&start));
        try {
            check_cuda(cudaEventCreate(&stop));
        } catch (...) {
            cudaEventDestroy(start);
            throw;
        }
    }

    ~EventPair() {
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
    }
};

enum class Implementation {
    Original,
    Workspace,
    Multiblock
};

double benchmark_implementation(
    Implementation implementation,
    const uint32_t* d_hist,
    uint32_t* d_hist_T,
    uint32_t* d_offsets,
    size_t num_blocks,
    size_t num_bins,
    uint32_t* d_workspace,
    size_t workspace_elements,
    int warmup_runs,
    int measured_runs
) {
    auto run = [&]() {
        cudaError_t result = cudaSuccess;

        switch (implementation) {
            case Implementation::Original:
                result = compute_global_offsets(
                    d_hist,
                    d_hist_T,
                    d_offsets,
                    num_blocks,
                    num_bins
                );
                break;

            case Implementation::Workspace:
                result = compute_global_offsets_with_workspace(
                    d_hist,
                    d_hist_T,
                    d_offsets,
                    num_blocks,
                    num_bins,
                    d_workspace,
                    workspace_elements
                );
                break;

            case Implementation::Multiblock:
                result = compute_global_offsets_multiblock(
                    d_hist,
                    d_hist_T,
                    d_offsets,
                    num_blocks,
                    num_bins,
                    d_workspace,
                    workspace_elements
                );
                break;
        }

        check_cuda(result);
    };

    // Warm up the GPU before measuring.
    for (int i = 0; i < warmup_runs; ++i) {
        run();
    }

    check_cuda(cudaDeviceSynchronize());

    EventPair events;

    // Preserve the original benchmark methodology:
    // time 100 consecutive executions and divide by 100.
    check_cuda(cudaEventRecord(events.start));

    for (int i = 0; i < measured_runs; ++i) {
        run();
    }

    check_cuda(cudaEventRecord(events.stop));
    check_cuda(cudaEventSynchronize(events.stop));

    float total_ms = 0.0f;

    check_cuda(cudaEventElapsedTime(
        &total_ms,
        events.start,
        events.stop
    ));

    return static_cast<double>(total_ms)
        / static_cast<double>(measured_runs);
}

void benchmark_case(
    size_t num_blocks,
    size_t num_bins,
    int warmup_runs,
    int measured_runs
) {
    const size_t n = num_blocks * num_bins;
    const size_t bytes = n * sizeof(uint32_t);

    // Identical input data for all implementations.
    std::vector<uint32_t> hist(n);

    for (size_t block = 0; block < num_blocks; ++block) {
        for (size_t bin = 0; bin < num_bins; ++bin) {
            hist[block * num_bins + bin] =
                static_cast<uint32_t>(
                    (block * 7 + bin * 3 + 1) % 11
                );
        }
    }

    const size_t workspace_elements =
        offset_scan::workspace_elements_required(n);

    // Allocate once, outside the measured section.
    DeviceBuffer d_hist(n);
    DeviceBuffer d_hist_T(n);
    DeviceBuffer d_offsets(n);
    DeviceBuffer d_workspace(workspace_elements);

    check_cuda(cudaMemcpy(
        d_hist.ptr,
        hist.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    const double original_ms = benchmark_implementation(
        Implementation::Original,
        d_hist.ptr,
        d_hist_T.ptr,
        d_offsets.ptr,
        num_blocks,
        num_bins,
        d_workspace.ptr,
        workspace_elements,
        warmup_runs,
        measured_runs
    );

    const double workspace_ms = benchmark_implementation(
        Implementation::Workspace,
        d_hist.ptr,
        d_hist_T.ptr,
        d_offsets.ptr,
        num_blocks,
        num_bins,
        d_workspace.ptr,
        workspace_elements,
        warmup_runs,
        measured_runs
    );

    const double multiblock_ms = benchmark_implementation(
        Implementation::Multiblock,
        d_hist.ptr,
        d_hist_T.ptr,
        d_offsets.ptr,
        num_blocks,
        num_bins,
        d_workspace.ptr,
        workspace_elements,
        warmup_runs,
        measured_runs
    );

    const double speedup =
        workspace_ms / multiblock_ms;

    std::cout
        << num_blocks << ","
        << num_bins << ","
        << n << ","
        << original_ms << ","
        << workspace_ms << ","
        << multiblock_ms << ","
        << speedup << "\n";
}

int main() {
    try {
        const int warmup_runs = 20;
        const int measured_runs = 100;

        const std::vector<std::pair<size_t, size_t>> shapes = {
            {32, 16},
            {64, 16},
            {256, 16},
            {625, 16},
            {4096, 16},
            {16384, 16},
            {65536, 16}
        };

        std::cout << std::fixed << std::setprecision(6);

        std::cout
            << "blocks,bins,entries,"
            << "original_avg_ms,"
            << "workspace_avg_ms,"
            << "multiblock_avg_ms,"
            << "multiblock_speedup\n";

        for (const auto& shape : shapes) {
            benchmark_case(
                shape.first,
                shape.second,
                warmup_runs,
                measured_runs
            );
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
