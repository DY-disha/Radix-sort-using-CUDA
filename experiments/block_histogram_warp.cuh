#pragma once
#include <cuda_runtime.h>
#include <cstddef>
// Same tile and output layout as histogram_device. Asynchronous on stream.
cudaError_t histogram_warp_device(const unsigned int* input, size_t n,
    unsigned shift, unsigned r, unsigned int* hist, cudaStream_t stream);
