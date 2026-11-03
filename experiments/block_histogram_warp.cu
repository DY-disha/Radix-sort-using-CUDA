#include "block_histogram_warp.cuh"
#include "radix_config.h"
#include <cstdint>
#include <stdexcept>

namespace {
constexpr unsigned B = radix_config::B;
constexpr unsigned Q = radix_config::Q;
constexpr unsigned WARPS = B / 32;
static_assert(B > 0 && B <= 1024 && B % 32 == 0);
static_assert(sizeof(unsigned int) == 4);
__global__ void histogram_warp_kernel(const unsigned int* input, size_t n,
    unsigned shift, unsigned H, unsigned int* hist) {
    extern __shared__ unsigned int counts[];
    for (unsigned i = threadIdx.x; i < WARPS * H; i += blockDim.x)
        counts[i] = 0;
    __syncthreads();

    const unsigned warp = threadIdx.x / 32;
    const size_t start = size_t(blockIdx.x) * B * Q;
    for (unsigned q = 0; q < Q; ++q) {
        const size_t i = start + size_t(q) * B + threadIdx.x;
        if (i < n) {
            const unsigned digit = (input[i] >> shift) & (H - 1);
            atomicAdd(&counts[warp * H + digit], 1u);
        }
    }
    __syncthreads();

    for (unsigned bin = threadIdx.x; bin < H; bin += blockDim.x) {
        unsigned int sum = 0;
        for (unsigned w = 0; w < WARPS; ++w)
            sum += counts[w * H + bin];
        hist[size_t(blockIdx.x) * H + bin] = sum;
    }
}
}
cudaError_t histogram_warp_device(const unsigned int* input, size_t n,
    unsigned shift, unsigned r, unsigned int* hist, cudaStream_t stream) {
    if (!(r == 1 || r == 2 || r == 4 || r == 8))
        throw std::invalid_argument("r must be 1, 2, 4, or 8");
    if (shift > 32 - r || shift % r != 0)
        throw std::invalid_argument("Invalid shift for this r");
    if (n > UINT32_MAX) throw std::overflow_error("N exceeds UINT32_MAX");
    if (!n) return cudaSuccess;
    if (!input || !hist || input == hist)
        throw std::invalid_argument("Use separate non-null buffers");
    const unsigned H = 1u << r;
    constexpr size_t tile = size_t(B) * Q;
    const size_t blocks = n / tile + (n % tile != 0);
    histogram_warp_kernel<<<unsigned(blocks), B, size_t(WARPS) * H * sizeof(unsigned int), stream>>>(
        input, n, shift, H, hist);
    return cudaGetLastError();
}
