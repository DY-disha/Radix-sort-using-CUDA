
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <climits>
#include "radix_config.h"
#include "scatter.cuh"

constexpr int THREADS_PER_BLOCK = radix_config::B;
constexpr int ITEMS_PER_THREAD = radix_config::Q;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
constexpr int NUM_BINS = radix_config::NUM_BINS;


// Extract the 8-bit digit currently being sorted
// shift = 0, 8, 16, or 24
__device__ __forceinline__
int get_digit(uint32_t value, int shift, uint32_t bin_mask) {
    return static_cast<int>((value >> shift) & bin_mask);
}


// Stable scatter kernel.
//
// input:
//   The input array for this radix pass
//
// output:
//   The output array for this radix pass - must not alias input
//
// offsets:
//   A flattened [OFFSET_NUM_BINS][num_blocks] array
//   offsets[bin * num_blocks + block] gives the global starting
//   output position for this bin in this block
//
// n:
//   Number of valid input elements
//
// num_blocks:
//   Number of input tiles, calculated from n and OFFSET_ITEMS_PER_BLOCK
//
// shift:
//   Which digit to process: 0, 8, 16, or 24
//
// Each rank is computed from three counts:
//   1. matching elements in earlier q iterations
//   2. matching elements in earlier warps in this q iteration
//   3. matching earlier lanes in this warp
//
// The first two counts use shared-memory histograms. The third count uses
// a warp ballot
__global__
void stable_scatter_kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ offsets,
    size_t n,
    int num_blocks,
    int shift,
    int items_per_block,
    uint32_t bin_mask) {
    const int block = static_cast<int>(blockIdx.x);

    // Block's first element in the input array
    const size_t tile_start =
        static_cast<size_t>(block) * items_per_block;

    // prior_counts contains all valid elements from earlier q iterations
    // warp_counts contains the current q iteration, one histogram per warp
    __shared__ uint32_t prior_counts[NUM_BINS];
    __shared__ uint32_t warp_counts[WARPS_PER_BLOCK][NUM_BINS];
    uint32_t* const warp_counts_flat = &warp_counts[0][0];

    for (int index = threadIdx.x; index < NUM_BINS; index += blockDim.x) {
        prior_counts[index] = 0;
    }

    for (int index = threadIdx.x; index < WARPS_PER_BLOCK * NUM_BINS; index += blockDim.x) {
        warp_counts_flat[index] = 0;
    }

    __syncthreads();

    // Each CUDA thread processes 22 elements, spaced 256
    // positions apart within the tile
    for (int q = 0; q < ITEMS_PER_THREAD; ++q) {
        const int local_index =
            threadIdx.x + q * blockDim.x;

        const size_t input_index =
            tile_start + local_index;

        // Clear the per-warp histograms for this q iteration
        for (int index = threadIdx.x; index < WARPS_PER_BLOCK * NUM_BINS; index += blockDim.x) {
            warp_counts_flat[index] = 0;
        }

        __syncthreads();

        const bool valid = input_index < n;
        const uint32_t value = valid ? input[input_index] : 0;
        const int digit = valid ? get_digit(value, shift, bin_mask) : 0;

        const int warp = threadIdx.x / WARP_SIZE;
        if (valid) {
            atomicAdd(&warp_counts[warp][digit], 1u);
        }

        // All warp histograms must be complete before they are read
        __syncthreads();

        uint32_t earlier_warp_rank = 0;
        if (valid) {
            for (int earlier_warp = 0; earlier_warp < warp; ++earlier_warp) {
                earlier_warp_rank += warp_counts[earlier_warp][digit];
            }
        }

        // Only earlier lanes in this warp contribute to this rank
        // All 32 lanes reach these collectives, including invalid tail lanes
        const unsigned active_mask = 0xffffffffu;
        const unsigned valid_mask = __ballot_sync(active_mask, valid);
        const unsigned matching_mask =__match_any_sync(active_mask, digit) & valid_mask;
        const unsigned lane = threadIdx.x & (WARP_SIZE - 1);
        const unsigned earlier_lane_mask = lane == 0 ? 0u : matching_mask & ((1u << lane) - 1u);
        const uint32_t earlier_lane_rank = valid ? static_cast<uint32_t>(__popc(earlier_lane_mask)) : 0;
        const uint32_t prior_rank = valid ? prior_counts[digit] : 0;

        // Prevent one thread from updating prior_counts before another
        // thread has taken its snapshot for this iteration
        __syncthreads();

        if (valid) {
            const uint32_t local_rank =
                prior_rank +
                earlier_warp_rank +
                earlier_lane_rank;

            // Global offsets for this bin and block, add local rank to get the final destination index
            const size_t offset_index = static_cast<size_t>(digit) * num_blocks + block;

            const uint32_t block_offset = offsets[offset_index];

            const size_t destination = static_cast<size_t>(block_offset) + local_rank;

            output[destination] = value;

            // Make this iteration visible to later q iterations only after
            // every thread has finished reading the prior counts
            atomicAdd(&prior_counts[digit], 1u);
        }

        __syncthreads();
    }
}


// Host-side launch helper
//
// The caller must already have allocated input, output, and offsets
// on the GPU, and must have populated offsets correctly
cudaError_t launch_stable_scatter(
    const uint32_t* input,
    uint32_t* output,
    const uint32_t* offsets,
    size_t n,
    int shift,
    cudaStream_t stream)
{
    if (shift != 0 && shift != 8 &&
        shift != 16 && shift != 24) {
        return cudaErrorInvalidValue;
    }

    if (n == 0)
        return cudaSuccess;

    if (!input || !output || !offsets || input == output ||
        output == offsets || n > UINT32_MAX)
        return cudaErrorInvalidValue;

    const size_t blocks = n / OFFSET_ITEMS_PER_BLOCK + (n % OFFSET_ITEMS_PER_BLOCK != 0);

    // The kernel and offset indexing use int for block count
    if (blocks > static_cast<size_t>(INT_MAX))
        return cudaErrorInvalidValue;

    const int num_blocks = static_cast<int>(blocks);

    stable_scatter_kernel<<<
        num_blocks, THREADS_PER_BLOCK, 0, stream
    >>>(
        input,
        output,
        offsets,
        n,
        num_blocks,
        shift,
        OFFSET_ITEMS_PER_BLOCK,
        static_cast<uint32_t>(OFFSET_NUM_BINS - 1)
    );

    // Checks whether the kernel launch itself failed
    return cudaGetLastError();
}
