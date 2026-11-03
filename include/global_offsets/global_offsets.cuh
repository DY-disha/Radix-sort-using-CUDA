
#pragma once

#include "transpose.cuh"
#include "scan.cuh"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

// ============================================================
// Global Offsets
// ============================================================
//
// Input:
//   d_hist[block * num_bins + bin]
//
// Intermediate:
//   d_hist_T[bin * num_blocks + block]
//
// Output:
//   d_offsets[bin * num_blocks + block]
//
// All three pointers refer to separate GPU arrays.
//
// Each array contains num_blocks * num_bins uint32_t entries.
// The total histogram count must fit in uint32_t.
//
// ============================================================
// 1. Original global-offset implementation
// ============================================================
//
// Uses a one-block transpose and the original hierarchical
// exclusive scan.
//

inline cudaError_t compute_global_offsets(
    const uint32_t* d_hist,
    uint32_t* d_hist_T,
    uint32_t* d_offsets,
    size_t num_blocks,
    size_t num_bins,
    cudaStream_t stream = 0
) {
    if (num_bins == 0) {
        throw std::invalid_argument(
            "num_bins must be positive"
        );
    }

    if (num_blocks >
        std::numeric_limits<size_t>::max() / num_bins) {
        throw std::overflow_error(
            "Histogram dimensions overflow size_t"
        );
    }

    const size_t n = num_blocks * num_bins;

    if (n == 0) {
        return cudaSuccess;
    }

    if (d_hist == nullptr ||
        d_hist_T == nullptr ||
        d_offsets == nullptr) {
        throw std::invalid_argument(
            "GPU pointers must not be null"
        );
    }

    if (d_hist == d_hist_T ||
        d_hist == d_offsets ||
        d_hist_T == d_offsets) {
        throw std::invalid_argument(
            "Use separate GPU arrays"
        );
    }

    // Stage 1: block-major -> bin-major.
    // Original single-block transpose.
    transpose_histogram<<<1, 256, 0, stream>>>(
        d_hist,
        d_hist_T,
        num_blocks,
        num_bins
    );

    cudaError_t result = cudaGetLastError();

    if (result != cudaSuccess) {
        return result;
    }

    // Stage 2: original hierarchical scan.
    return offset_scan::exclusive_scan(
        d_hist_T,
        d_offsets,
        n,
        stream
    );
}


// ============================================================
// 2. Workspace-based optimized implementation
// ============================================================
//
// Uses caller-provided temporary GPU memory so that
// cudaMalloc/cudaFree are not performed inside the scan.
//
// The transpose still uses one CUDA block.
//
// Required workspace:
//
//   offset_scan::workspace_elements_required(n)
//

inline cudaError_t compute_global_offsets_with_workspace(
    const uint32_t* d_hist,
    uint32_t* d_hist_T,
    uint32_t* d_offsets,
    size_t num_blocks,
    size_t num_bins,
    uint32_t* d_workspace,
    size_t workspace_elements,
    cudaStream_t stream = 0
) {
    if (num_bins == 0) {
        throw std::invalid_argument(
            "num_bins must be positive"
        );
    }

    if (num_blocks >
        std::numeric_limits<size_t>::max() / num_bins) {
        throw std::overflow_error(
            "Histogram dimensions overflow size_t"
        );
    }

    const size_t n = num_blocks * num_bins;

    if (n == 0) {
        return cudaSuccess;
    }

    if (d_hist == nullptr ||
        d_hist_T == nullptr ||
        d_offsets == nullptr) {
        throw std::invalid_argument(
            "GPU pointers must not be null"
        );
    }

    if (d_hist == d_hist_T ||
        d_hist == d_offsets ||
        d_hist_T == d_offsets) {
        throw std::invalid_argument(
            "Use separate GPU arrays"
        );
    }

    const size_t required_workspace =
        offset_scan::workspace_elements_required(n);

    if (required_workspace > 0) {
        if (d_workspace == nullptr) {
            throw std::invalid_argument(
                "Workspace pointer must not be null"
            );
        }

        if (workspace_elements < required_workspace) {
            throw std::invalid_argument(
                "Insufficient workspace"
            );
        }
    }

    // Stage 1: original single-block transpose.
    transpose_histogram<<<1, 256, 0, stream>>>(
        d_hist,
        d_hist_T,
        num_blocks,
        num_bins
    );

    cudaError_t result = cudaGetLastError();

    if (result != cudaSuccess) {
        return result;
    }

    // Stage 2: workspace-based hierarchical scan.
    return offset_scan::exclusive_scan_with_workspace(
        d_hist_T,
        d_offsets,
        n,
        d_workspace,
        workspace_elements,
        stream
    );
}


// ============================================================
// 3. Compatibility wrapper for small inputs
// ============================================================
//
// Retained so existing tests/code using
// compute_global_offsets_small() continue to compile.
//
// Preserves the original limit of 512 histogram entries.
//

inline cudaError_t compute_global_offsets_small(
    const uint32_t* d_hist,
    uint32_t* d_hist_T,
    uint32_t* d_offsets,
    size_t num_blocks,
    size_t num_bins,
    cudaStream_t stream = 0
) {
    if (num_bins == 0) {
        throw std::invalid_argument(
            "num_bins must be positive"
        );
    }

    if (num_blocks >
        offset_scan::TILE_SIZE / num_bins) {
        throw std::invalid_argument(
            "compute_global_offsets_small supports at most "
            "512 histogram entries"
        );
    }

    return compute_global_offsets(
        d_hist,
        d_hist_T,
        d_offsets,
        num_blocks,
        num_bins,
        stream
    );
}


// ============================================================
// 4. Experimental multi-block transpose implementation
// ============================================================
//
// Offset tuning experiment.
//
// Compared with compute_global_offsets_with_workspace():
//
// Original:
//   Transpose: 1 CUDA block, 256 threads
//   Scan:      reusable workspace
//
// Experimental:
//   Transpose: multiple CUDA blocks, 256 threads each
//   Scan:      reusable workspace
//
// The transpose kernel itself is unchanged because it already
// supports arbitrary grid dimensions through a grid-stride loop.
//
// This version is kept separate until correctness and
// performance have been tested.
// ============================================================

inline cudaError_t compute_global_offsets_multiblock(
    const uint32_t* d_hist,
    uint32_t* d_hist_T,
    uint32_t* d_offsets,
    size_t num_blocks,
    size_t num_bins,
    uint32_t* d_workspace,
    size_t workspace_elements,
    cudaStream_t stream = 0
) {
    if (num_bins == 0) {
        throw std::invalid_argument(
            "num_bins must be positive"
        );
    }

    // Check multiplication overflow.
    if (num_blocks >
        std::numeric_limits<size_t>::max() / num_bins) {
        throw std::overflow_error(
            "Histogram dimensions overflow size_t"
        );
    }

    const size_t n = num_blocks * num_bins;

    if (n == 0) {
        return cudaSuccess;
    }

    // Validate GPU input/output pointers.
    if (d_hist == nullptr ||
        d_hist_T == nullptr ||
        d_offsets == nullptr) {
        throw std::invalid_argument(
            "GPU pointers must not be null"
        );
    }

    // The three arrays must be separate.
    if (d_hist == d_hist_T ||
        d_hist == d_offsets ||
        d_hist_T == d_offsets) {
        throw std::invalid_argument(
            "Use separate GPU arrays"
        );
    }

    // Validate workspace capacity.
    const size_t required_workspace =
        offset_scan::workspace_elements_required(n);

    if (required_workspace > 0) {
        if (d_workspace == nullptr) {
            throw std::invalid_argument(
                "Workspace pointer must not be null"
            );
        }

        if (workspace_elements < required_workspace) {
            throw std::invalid_argument(
                "Insufficient workspace"
            );
        }
    }

    // --------------------------------------------------------
    // Stage 1: multi-block histogram transpose
    // --------------------------------------------------------

    constexpr unsigned THREADS = 256;

    // Conservative upper bound on launched blocks.
    // The grid-stride loop handles any remaining entries.
    constexpr size_t MAX_BLOCKS = 65535;

    // ceil(n / THREADS), avoiding overflow from n + THREADS - 1.
    const size_t required_blocks =
        1 + (n - 1) / THREADS;

    const unsigned grid_blocks =
        static_cast<unsigned>(
            required_blocks < MAX_BLOCKS
                ? required_blocks
                : MAX_BLOCKS
        );

    transpose_histogram<<<grid_blocks, THREADS, 0, stream>>>(
        d_hist,
        d_hist_T,
        num_blocks,
        num_bins
    );

    cudaError_t result = cudaGetLastError();

    if (result != cudaSuccess) {
        return result;
    }

    // --------------------------------------------------------
    // Stage 2: workspace-based hierarchical exclusive scan
    // --------------------------------------------------------

    return offset_scan::exclusive_scan_with_workspace(
        d_hist_T,
        d_offsets,
        n,
        d_workspace,
        workspace_elements,
        stream
    );
}
