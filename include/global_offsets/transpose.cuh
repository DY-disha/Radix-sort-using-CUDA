#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Input:  hist[block * num_bins + bin]
// Output: hist_T[bin * num_blocks + block]
__global__ void transpose_histogram(
    const uint32_t* hist,
    uint32_t* hist_T,
    size_t num_blocks,
    size_t num_bins
) {
    const size_t total_entries = num_blocks * num_bins;

    const size_t thread_index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t index = thread_index;
         index < total_entries;
         index += stride) {

        // Recover the input row and column.
        const size_t block = index / num_bins;
        const size_t bin = index % num_bins;

        // Write the count to its transposed position.
        hist_T[bin * num_blocks + block] = hist[index];
    }
}