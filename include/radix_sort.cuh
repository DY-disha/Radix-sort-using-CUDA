#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Owns histogram, transpose, offset and scan scratch buffers for exactly n keys.
// Allocate once outside timing. Do not share concurrently between streams.
class RadixWorkspace {
public:
    explicit RadixWorkspace(size_t n);
    ~RadixWorkspace();
    RadixWorkspace(const RadixWorkspace&) = delete;
    RadixWorkspace& operator=(const RadixWorkspace&) = delete;
    size_t size() const { return n_; }
private:
    size_t n_, blocks_, scan_elements_;
    uint32_t *hist_ = nullptr, *transposed_ = nullptr;
    uint32_t *offsets_ = nullptr, *scan_ = nullptr;
    void release() noexcept;
    friend void radix_pass_device(const uint32_t*, uint32_t*, size_t,
                                  unsigned, RadixWorkspace&, cudaStream_t);
};
// GPU buffers must be separate, non-overlapping and contain n uint32_t entries.
// Calls throw on invalid arguments or CUDA launch errors. Synchronize stream
// before reading results, reusing workspace on another stream, or freeing buffers.
void radix_pass_device(const uint32_t* input, uint32_t* output, size_t n,
                       unsigned shift, RadixWorkspace&, cudaStream_t stream = 0);
// Sort ascending unsigned 32-bit keys, all bits. Returns the final device pointer.
// Four passes return to input; both input and scratch are modified.
uint32_t* radix_sort_device(uint32_t* input, uint32_t* scratch, size_t n,
                           RadixWorkspace&, cudaStream_t stream = 0);
