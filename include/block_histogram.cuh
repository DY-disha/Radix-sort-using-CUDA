#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>
unsigned threads_per_block();
unsigned elements_per_thread();
size_t tile_size();
size_t block_count(size_t n);
std::vector<unsigned int> histogram_cpu(const std::vector<unsigned int>&, unsigned, unsigned);
std::vector<unsigned int> histogram_gpu(const std::vector<unsigned int>&, unsigned, unsigned);
void check_sums(const std::vector<unsigned int>&, size_t, unsigned);
// May throw for invalid arguments; execution is asynchronous on stream.
cudaError_t histogram_device(const unsigned int*, size_t, unsigned, unsigned,
                             unsigned int*, cudaStream_t);
