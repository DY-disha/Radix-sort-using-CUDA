#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace offset_scan {

constexpr unsigned THREADS = 256;
constexpr unsigned TILE_SIZE = 512;


// ============================================================
// Single-block exclusive scan
// ============================================================
//
// Handles at most TILE_SIZE = 512 entries.
// Each CUDA thread processes two values.
//
__global__ void scan_single_block(
    const uint32_t* input,
    uint32_t* output,
    size_t n
) {
    __shared__ uint32_t values[TILE_SIZE];

    const unsigned t = threadIdx.x;

    values[t] =
        t < n
            ? input[t]
            : 0;

    values[t + THREADS] =
        t + THREADS < n
            ? input[t + THREADS]
            : 0;

    __syncthreads();

    // Upsweep.
    for (unsigned stride = 1;
         stride < TILE_SIZE;
         stride *= 2) {

        const unsigned index =
            (t + 1) * 2 * stride - 1;

        if (index < TILE_SIZE) {
            values[index] += values[index - stride];
        }

        __syncthreads();
    }

    // Root becomes zero for exclusive scan.
    if (t == 0) {
        values[TILE_SIZE - 1] = 0;
    }

    __syncthreads();

    // Downsweep.
    for (unsigned stride = TILE_SIZE / 2;
         stride > 0;
         stride /= 2) {

        const unsigned index =
            (t + 1) * 2 * stride - 1;

        if (index < TILE_SIZE) {
            const uint32_t left_sum =
                values[index - stride];

            values[index - stride] =
                values[index];

            values[index] += left_sum;
        }

        __syncthreads();
    }

    if (t < n) {
        output[t] = values[t];
    }

    if (t + THREADS < n) {
        output[t + THREADS] =
            values[t + THREADS];
    }
}


// ============================================================
// Multi-block tile scan
// ============================================================
//
// Each CUDA block scans one tile of up to 512 entries.
// block_sums receives the total sum of each tile.
//
__global__ void scan_tiles(
    const uint32_t* input,
    uint32_t* output,
    uint32_t* block_sums,
    size_t n
) {
    __shared__ uint32_t values[TILE_SIZE];

    const unsigned t = threadIdx.x;

    const size_t tile_start =
        static_cast<size_t>(blockIdx.x) * TILE_SIZE;

    const size_t index0 =
        tile_start + t;

    const size_t index1 =
        tile_start + t + THREADS;

    values[t] =
        index0 < n
            ? input[index0]
            : 0;

    values[t + THREADS] =
        index1 < n
            ? input[index1]
            : 0;

    __syncthreads();

    // Upsweep.
    for (unsigned stride = 1;
         stride < TILE_SIZE;
         stride *= 2) {

        const unsigned index =
            (t + 1) * 2 * stride - 1;

        if (index < TILE_SIZE) {
            values[index] += values[index - stride];
        }

        __syncthreads();
    }

    // Save the total before clearing the root.
    if (t == 0) {
        block_sums[blockIdx.x] =
            values[TILE_SIZE - 1];

        values[TILE_SIZE - 1] = 0;
    }

    __syncthreads();

    // Downsweep.
    for (unsigned stride = TILE_SIZE / 2;
         stride > 0;
         stride /= 2) {

        const unsigned index =
            (t + 1) * 2 * stride - 1;

        if (index < TILE_SIZE) {
            const uint32_t left_sum =
                values[index - stride];

            values[index - stride] =
                values[index];

            values[index] += left_sum;
        }

        __syncthreads();
    }

    if (index0 < n) {
        output[index0] =
            values[t];
    }

    if (index1 < n) {
        output[index1] =
            values[t + THREADS];
    }
}


// ============================================================
// Add scanned block offsets
// ============================================================
//
// Each locally scanned tile receives the prefix sum of all
// preceding tiles.
//
__global__ void add_block_offsets(
    uint32_t* output,
    const uint32_t* block_offsets,
    size_t n
) {
    const size_t tile_start =
        static_cast<size_t>(blockIdx.x) * TILE_SIZE;

    const size_t index0 =
        tile_start + threadIdx.x;

    const size_t index1 =
        tile_start + threadIdx.x + THREADS;

    const uint32_t offset =
        block_offsets[blockIdx.x];

    if (index0 < n) {
        output[index0] += offset;
    }

    if (index1 < n) {
        output[index1] += offset;
    }
}


// ============================================================
// Small scan interface
// ============================================================
//
// Supports at most 512 entries.
//
inline cudaError_t exclusive_scan_small(
    const uint32_t* input,
    uint32_t* output,
    size_t n,
    cudaStream_t stream = 0
) {
    if (n > TILE_SIZE) {
        throw std::invalid_argument(
            "exclusive_scan_small supports at most 512 entries"
        );
    }

    if (n == 0) {
        return cudaSuccess;
    }

    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument(
            "GPU pointers must not be null"
        );
    }

    scan_single_block<<<1, THREADS, 0, stream>>>(
        input,
        output,
        n
    );

    return cudaGetLastError();
}


// ============================================================
// Original general multi-block scan
// ============================================================
//
// This is the original implementation.
//
// It allocates temporary GPU memory internally. Keep it so that
// the original implementation remains available for correctness
// testing and before/after performance comparison.
//
inline cudaError_t exclusive_scan(
    const uint32_t* input,
    uint32_t* output,
    size_t n,
    cudaStream_t stream = 0
) {
    if (n == 0) {
        return cudaSuccess;
    }

    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument(
            "GPU pointers must not be null"
        );
    }

    if (n <= TILE_SIZE) {
        return exclusive_scan_small(
            input,
            output,
            n,
            stream
        );
    }

    const size_t num_blocks =
        1 + (n - 1) / TILE_SIZE;

    if (num_blocks >
        std::numeric_limits<unsigned>::max()) {

        return cudaErrorInvalidConfiguration;
    }

    uint32_t* block_sums = nullptr;
    uint32_t* block_offsets = nullptr;

    const size_t block_bytes =
        num_blocks * sizeof(uint32_t);

    cudaError_t result =
        cudaMalloc(
            reinterpret_cast<void**>(&block_sums),
            block_bytes
        );

    if (result != cudaSuccess) {
        return result;
    }

    result =
        cudaMalloc(
            reinterpret_cast<void**>(&block_offsets),
            block_bytes
        );

    if (result != cudaSuccess) {
        cudaFree(block_sums);
        return result;
    }

    // Stage 1: scan each tile.
    scan_tiles<<<
        static_cast<unsigned>(num_blocks),
        THREADS,
        0,
        stream
    >>>(
        input,
        output,
        block_sums,
        n
    );

    result = cudaGetLastError();

    if (result != cudaSuccess) {
        cudaFree(block_offsets);
        cudaFree(block_sums);
        return result;
    }

    // Stage 2: recursively scan tile totals.
    result =
        exclusive_scan(
            block_sums,
            block_offsets,
            num_blocks,
            stream
        );

    if (result != cudaSuccess) {
        cudaFree(block_offsets);
        cudaFree(block_sums);
        return result;
    }

    // Stage 3: add tile prefixes.
    add_block_offsets<<<
        static_cast<unsigned>(num_blocks),
        THREADS,
        0,
        stream
    >>>(
        output,
        block_offsets,
        n
    );

    result = cudaGetLastError();

    if (result != cudaSuccess) {
        cudaFree(block_offsets);
        cudaFree(block_sums);
        return result;
    }

    result = cudaFree(block_offsets);

    if (result != cudaSuccess) {
        cudaFree(block_sums);
        return result;
    }

    return cudaFree(block_sums);
}


// ============================================================
// Workspace-size calculation
// ============================================================
//
// Returns the number of uint32_t elements required by the
// allocation-free hierarchical scan.
//
// At every hierarchy level we need:
//
//     num_blocks values for block sums
//     num_blocks values for scanned block offsets
//
// The same process is repeated recursively for the block sums.
//
inline size_t workspace_elements_required(
    size_t n
) {
    size_t total = 0;

    while (n > TILE_SIZE) {
        const size_t num_blocks =
            1 + (n - 1) / TILE_SIZE;

        if (num_blocks >
            std::numeric_limits<size_t>::max() / 2) {

            throw std::overflow_error(
                "Scan workspace size overflow"
            );
        }

        const size_t level_elements =
            2 * num_blocks;

        if (total >
            std::numeric_limits<size_t>::max()
            - level_elements) {

            throw std::overflow_error(
                "Scan workspace size overflow"
            );
        }

        total += level_elements;

        n = num_blocks;
    }

    return total;
}


// Return workspace requirement in bytes.
inline size_t workspace_bytes_required(
    size_t n
) {
    const size_t elements =
        workspace_elements_required(n);

    if (elements >
        std::numeric_limits<size_t>::max()
        / sizeof(uint32_t)) {

        throw std::overflow_error(
            "Scan workspace byte size overflow"
        );
    }

    return elements * sizeof(uint32_t);
}


// ============================================================
// Workspace-based exclusive scan
// ============================================================
//
// Same hierarchical algorithm as exclusive_scan(), but no
// cudaMalloc/cudaFree calls occur inside this function.
//
// workspace must point to GPU memory containing at least
// workspace_elements_required(n) uint32_t elements.
//
inline cudaError_t exclusive_scan_with_workspace(
    const uint32_t* input,
    uint32_t* output,
    size_t n,
    uint32_t* workspace,
    size_t workspace_elements,
    cudaStream_t stream = 0
) {
    if (n == 0) {
        return cudaSuccess;
    }

    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument(
            "GPU pointers must not be null"
        );
    }

    // Single-block case requires no temporary workspace.
    if (n <= TILE_SIZE) {
        return exclusive_scan_small(
            input,
            output,
            n,
            stream
        );
    }

    const size_t required =
        workspace_elements_required(n);

    if (workspace == nullptr) {
        throw std::invalid_argument(
            "Workspace pointer must not be null"
        );
    }

    if (workspace_elements < required) {
        throw std::invalid_argument(
            "Insufficient scan workspace"
        );
    }

    const size_t num_blocks =
        1 + (n - 1) / TILE_SIZE;

    if (num_blocks >
        std::numeric_limits<unsigned>::max()) {

        return cudaErrorInvalidConfiguration;
    }

    // Workspace layout for this hierarchy level:
    //
    // [ block_sums | block_offsets | child workspace ... ]
    //
    uint32_t* block_sums =
        workspace;

    uint32_t* block_offsets =
        workspace + num_blocks;

    uint32_t* child_workspace =
        workspace + 2 * num_blocks;

    const size_t child_workspace_elements =
        workspace_elements - 2 * num_blocks;

    // --------------------------------------------------------
    // Stage 1:
    // Scan every tile independently.
    // --------------------------------------------------------

    scan_tiles<<<
        static_cast<unsigned>(num_blocks),
        THREADS,
        0,
        stream
    >>>(
        input,
        output,
        block_sums,
        n
    );

    cudaError_t result =
        cudaGetLastError();

    if (result != cudaSuccess) {
        return result;
    }

    // --------------------------------------------------------
    // Stage 2:
    // Recursively scan tile totals using the remaining
    // caller-provided workspace.
    // --------------------------------------------------------

    result =
        exclusive_scan_with_workspace(
            block_sums,
            block_offsets,
            num_blocks,
            child_workspace,
            child_workspace_elements,
            stream
        );

    if (result != cudaSuccess) {
        return result;
    }

    // --------------------------------------------------------
    // Stage 3:
    // Add the scanned tile prefix to each local tile.
    // --------------------------------------------------------

    add_block_offsets<<<
        static_cast<unsigned>(num_blocks),
        THREADS,
        0,
        stream
    >>>(
        output,
        block_offsets,
        n
    );

    return cudaGetLastError();
}

} // namespace offset_scan