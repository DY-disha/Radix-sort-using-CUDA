
#include "global_offsets/global_offsets.cuh"
#include "offset_reference.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

void check_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

// Release GPU allocations automatically, even if a test fails.
struct DeviceBuffer {
    uint32_t* ptr = nullptr;

    explicit DeviceBuffer(size_t elements) {
        if (elements != 0) {
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

void compare_result(
    const std::vector<uint32_t>& actual,
    const std::vector<uint32_t>& expected,
    const std::string& name,
    const std::string& implementation
) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(
            name + " [" + implementation + "]: size mismatch"
        );
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(
                name + " [" + implementation + "]"
                + ": mismatch at output index "
                + std::to_string(i)
                + ", expected "
                + std::to_string(expected[i])
                + ", got "
                + std::to_string(actual[i])
            );
        }
    }
}

void test_offsets(
    const std::vector<uint32_t>& hist,
    size_t num_blocks,
    size_t num_bins,
    bool print_output = false
) {
    const auto expected =
        compute_offsets_cpu(hist, num_blocks, num_bins);

    const std::string name =
        "blocks=" + std::to_string(num_blocks)
        + ", bins=" + std::to_string(num_bins);

    // --------------------------------------------------------
    // Handle empty histograms
    // --------------------------------------------------------

    if (hist.empty()) {
        check_cuda(
            compute_global_offsets(
                nullptr,
                nullptr,
                nullptr,
                num_blocks,
                num_bins
            )
        );

        check_cuda(
            compute_global_offsets_with_workspace(
                nullptr,
                nullptr,
                nullptr,
                num_blocks,
                num_bins,
                nullptr,
                0
            )
        );

        check_cuda(
            compute_global_offsets_multiblock(
                nullptr,
                nullptr,
                nullptr,
                num_blocks,
                num_bins,
                nullptr,
                0
            )
        );

        return;
    }

    const size_t n = hist.size();
    const size_t bytes = n * sizeof(uint32_t);

    const size_t workspace_elements =
        offset_scan::workspace_elements_required(n);

    // --------------------------------------------------------
    // Allocate GPU arrays
    // --------------------------------------------------------

    DeviceBuffer d_hist(n);

    // Original implementation
    DeviceBuffer d_hist_T_original(n);
    DeviceBuffer d_offsets_original(n);

    // Workspace implementation
    DeviceBuffer d_hist_T_workspace(n);
    DeviceBuffer d_offsets_workspace(n);

    // Multi-block implementation
    DeviceBuffer d_hist_T_multiblock(n);
    DeviceBuffer d_offsets_multiblock(n);

    // Shared reusable scan workspace
    DeviceBuffer d_workspace(workspace_elements);

    // --------------------------------------------------------
    // Copy input histogram to GPU
    // --------------------------------------------------------

    check_cuda(
        cudaMemcpy(
            d_hist.ptr,
            hist.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );

    // --------------------------------------------------------
    // Implementation 1: Original
    // --------------------------------------------------------

    check_cuda(
        compute_global_offsets(
            d_hist.ptr,
            d_hist_T_original.ptr,
            d_offsets_original.ptr,
            num_blocks,
            num_bins
        )
    );

    // --------------------------------------------------------
    // Implementation 2: Workspace optimized
    // --------------------------------------------------------

    check_cuda(
        compute_global_offsets_with_workspace(
            d_hist.ptr,
            d_hist_T_workspace.ptr,
            d_offsets_workspace.ptr,
            num_blocks,
            num_bins,
            d_workspace.ptr,
            workspace_elements
        )
    );

    // --------------------------------------------------------
    // Implementation 3: Multi-block optimized
    // --------------------------------------------------------

    check_cuda(
        compute_global_offsets_multiblock(
            d_hist.ptr,
            d_hist_T_multiblock.ptr,
            d_offsets_multiblock.ptr,
            num_blocks,
            num_bins,
            d_workspace.ptr,
            workspace_elements
        )
    );

    // Wait until all GPU operations finish.
    check_cuda(cudaDeviceSynchronize());

    // --------------------------------------------------------
    // Copy all three results back to CPU
    // --------------------------------------------------------

    std::vector<uint32_t> actual_original(n);
    std::vector<uint32_t> actual_workspace(n);
    std::vector<uint32_t> actual_multiblock(n);

    check_cuda(
        cudaMemcpy(
            actual_original.data(),
            d_offsets_original.ptr,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );

    check_cuda(
        cudaMemcpy(
            actual_workspace.data(),
            d_offsets_workspace.ptr,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );

    check_cuda(
        cudaMemcpy(
            actual_multiblock.data(),
            d_offsets_multiblock.ptr,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );

    // --------------------------------------------------------
    // Compare against CPU reference
    // --------------------------------------------------------

    compare_result(
        actual_original,
        expected,
        name,
        "original"
    );

    compare_result(
        actual_workspace,
        expected,
        name,
        "workspace"
    );

    compare_result(
        actual_multiblock,
        expected,
        name,
        "multiblock"
    );

    // --------------------------------------------------------
    // Compare GPU implementations with one another
    // --------------------------------------------------------

    compare_result(
        actual_original,
        actual_workspace,
        name,
        "original vs workspace"
    );

    compare_result(
        actual_multiblock,
        actual_workspace,
        name,
        "multiblock vs workspace"
    );

    // --------------------------------------------------------
    // Optional output for the hand-checked example
    // --------------------------------------------------------

    if (print_output) {
        std::cout << "GPU offsets (rows are bins):\n";

        for (size_t bin = 0; bin < num_bins; ++bin) {
            std::cout << "Bin " << bin << ": ";

            for (size_t block = 0; block < num_blocks; ++block) {
                std::cout
                    << actual_multiblock[
                        bin * num_blocks + block
                    ]
                    << " ";
            }

            std::cout << "\n";
        }
    }
}

int main() {
    try {
        // ----------------------------------------------------
        // Test 1: Hand-checked 3-block, 4-bin example
        // ----------------------------------------------------

        test_offsets(
            {
                2, 1, 1, 0,
                1, 2, 1, 0,
                0, 1, 1, 2
            },
            3,
            4,
            true
        );

        std::cout
            << "PASS: 3-block, 4-bin example "
            << "[original + workspace + multiblock]\n";

        size_t passed = 1;

        // ----------------------------------------------------
        // Additional test shapes
        // ----------------------------------------------------

        const std::vector<std::pair<size_t, size_t>> shapes = {
            {0, 4},
            {1, 1},
            {1, 4},
            {3, 4},
            {7, 16},
            {16, 16},
            {17, 16},
            {31, 16},
            {32, 16},
            {2, 256},
            {27, 19},      // 513 entries
            {33, 16},      // 528 entries
            {64, 16},      // 1024 entries
            {41, 25},      // 1025 entries
            {256, 16},     // 4096 entries
            {625, 16},      // 10000 entries
            {4096, 16},     // 65536 entries
            {16384, 16},    // 262144 entries
            {65536, 16}     // 1048576 entries
        };

        // ----------------------------------------------------
        // Test different histogram distributions
        // ----------------------------------------------------

        for (const auto& shape : shapes) {
            const size_t blocks = shape.first;
            const size_t bins = shape.second;

            for (unsigned mode = 0; mode < 4; ++mode) {
                std::vector<uint32_t> hist(
                    blocks * bins,
                    0
                );

                for (size_t block = 0;
                     block < blocks;
                     ++block) {

                    for (size_t bin = 0;
                         bin < bins;
                         ++bin) {

                        uint32_t count = 0;

                        // Mode 0: All histogram counts are zero.

                        if (mode == 1) {
                            // Mode 1: Counts in first bin.
                            count = bin == 0 ? 7 : 0;

                        } else if (mode == 2) {
                            // Mode 2: Counts in last bin.
                            count = bin == bins - 1 ? 7 : 0;

                        } else if (mode == 3) {
                            // Mode 3: Mixed counts.
                            count = static_cast<uint32_t>(
                                (block * 7 + bin * 3 + 1) % 11
                            );
                        }

                        hist[block * bins + bin] = count;
                    }
                }

                test_offsets(
                    hist,
                    blocks,
                    bins
                );

                ++passed;

                std::cout
                    << "PASS: blocks=" << blocks
                    << ", bins=" << bins
                    << ", n=" << blocks * bins
                    << ", mode=" << mode
                    << " [original + workspace + multiblock]\n";
            }
        }

        // ----------------------------------------------------
        // Final summary
        // ----------------------------------------------------

        std::cout
            << "PASS: All "
            << passed
            << " global-offset cases matched CPU reference "
            << "for all three implementations.\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "ERROR: "
            << error.what()
            << "\n";

        return 1;
    }
}
