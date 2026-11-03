
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

cudaError_t launch_stable_scatter(
    const uint32_t* input,
    uint32_t* output,
    const uint32_t* offsets,
    size_t n,
    int shift,
    cudaStream_t stream = nullptr
);
