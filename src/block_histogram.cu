#include "block_histogram.cuh"
#include "radix_config.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using Key = unsigned int;
static_assert(sizeof(Key) == 4, "Requires 32-bit unsigned int");

// Initial configuration.
constexpr unsigned B = radix_config::B;
constexpr unsigned Q = radix_config::Q;
constexpr size_t TILE = size_t(B) * Q;

// Allow main.cu to read the configuration.
unsigned threads_per_block() { return B; }
unsigned elements_per_thread() { return Q; }
size_t tile_size() { return TILE; }

void cuda_check(cudaError_t result)
{
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

void check_pass(unsigned shift, unsigned r)
{
    if (!(r == 1 || r == 2 || r == 4 || r == 8)) {
        throw std::invalid_argument("r must be 1, 2, 4, or 8");
    }

    if (shift > 32 - r || shift % r != 0) {
        throw std::invalid_argument("Invalid shift for this r");
    }
}

size_t block_count(size_t n)
{
    return n / TILE + (n % TILE != 0);
}

// Sequential reference: one histogram per tile.
std::vector<Key> histogram_cpu(
    const std::vector<Key>& input,
    unsigned shift,
    unsigned r)
{
    check_pass(shift, r);

    unsigned H = 1u << r;
    size_t G = block_count(input.size());

    std::vector<Key> hist(G * H, 0);

    for (size_t i = 0; i < input.size(); ++i) {
        size_t block = i / TILE;
        unsigned digit = (input[i] >> shift) & (H - 1);

        ++hist[block * H + digit];
    }

    return hist;
}

// GPU kernel: one block processes one tile.
__global__ void block_histogram_kernel(
    const Key* input,
    size_t n,
    unsigned shift,
    unsigned H,
    Key* hist)
{
    extern __shared__ Key counts[];

    // Step 1: initialize this block's histogram.
    for (unsigned bin = threadIdx.x;
         bin < H;
         bin += blockDim.x) {
        counts[bin] = 0;
    }

    __syncthreads();

    // Step 2: count this block's input elements.
    size_t start = size_t(blockIdx.x) * blockDim.x * Q;

    for (unsigned j = 0; j < Q; ++j) {
        size_t i =
            start + size_t(j) * blockDim.x + threadIdx.x;

        if (i < n) {
            unsigned digit = (input[i] >> shift) & (H - 1);
            atomicAdd(&counts[digit], 1u);
        }
    }

    __syncthreads();

    // Step 3: store the completed histogram.
    for (unsigned bin = threadIdx.x;
         bin < H;
         bin += blockDim.x) {
        hist[size_t(blockIdx.x) * H + bin] = counts[bin];
    }
}

// Test wrapper: uploads input, runs the kernel,
// and returns the histogram to CPU memory.
std::vector<Key> histogram_gpu(
    const std::vector<Key>& input,
    unsigned shift,
    unsigned r)
{
    check_pass(shift, r);

    unsigned H = 1u << r;
    size_t G = block_count(input.size());

    std::vector<Key> result(G * H);

    if (input.empty()) {
        return result;
    }

    Key* d_input = nullptr;
    Key* d_hist = nullptr;

    try {
        cuda_check(cudaMalloc(
            reinterpret_cast<void**>(&d_input),
            input.size() * sizeof(Key)));

        cuda_check(cudaMalloc(
            reinterpret_cast<void**>(&d_hist),
            result.size() * sizeof(Key)));

        cuda_check(cudaMemcpy(
            d_input,
            input.data(),
            input.size() * sizeof(Key),
            cudaMemcpyHostToDevice));

        block_histogram_kernel<<<
            static_cast<unsigned>(G), B, H * sizeof(Key)>>>(
                d_input, input.size(), shift, H, d_hist);

        cuda_check(cudaGetLastError());
        cuda_check(cudaDeviceSynchronize());

        cuda_check(cudaMemcpy(
            result.data(),
            d_hist,
            result.size() * sizeof(Key),
            cudaMemcpyDeviceToHost));
    }
    catch (...) {
        cudaFree(d_input);
        cudaFree(d_hist);
        throw;
    }

    cuda_check(cudaFree(d_input));
    cuda_check(cudaFree(d_hist));

    return result;
}

// Check that each tile counts every valid element exactly once.
void check_sums(
    const std::vector<Key>& hist,
    size_t n,
    unsigned H)
{
    uint64_t total = 0;

    for (size_t block = 0; block < block_count(n); ++block) {
        uint64_t sum = 0;

        for (unsigned bin = 0; bin < H; ++bin) {
            sum += hist[block * H + bin];
        }

        size_t remaining = n - block * TILE;
        size_t expected = remaining < TILE ? remaining : TILE;

        if (sum != expected) {
            throw std::runtime_error("Incorrect tile sum");
        }

        total += sum;
    }

    if (total != n) {
        throw std::runtime_error("Incorrect total sum");
    }
}
// Integration entry point. Caller owns GPU input/output buffers.
// Output has block_count(n) * (1u << r) entries in block-major order.
// No allocation, CPU copies, synchronization, or freeing occurs here.
// Queue subsequent offset operations on the same stream.
cudaError_t histogram_device(const Key* d_input, size_t n,
                             unsigned shift, unsigned r, Key* d_hist,
                             cudaStream_t stream)
{
    check_pass(shift, r);
    if (n > UINT32_MAX)
        throw std::overflow_error("Combined pipeline requires N <= UINT32_MAX");
    if (n == 0) return cudaSuccess;
    if (!d_input || !d_hist || d_input == d_hist)
        throw std::invalid_argument("Use separate non-null GPU input and histogram buffers");
    const unsigned H = 1u << r;
    block_histogram_kernel<<<static_cast<unsigned>(block_count(n)), B,
                             H * sizeof(Key), stream>>>(d_input, n, shift, H, d_hist);
    return cudaGetLastError();
}
