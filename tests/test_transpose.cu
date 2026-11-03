#include "global_offsets/transpose.cuh"

#include <iostream>
#include <stdexcept>
#include <vector>

void check_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

int main() {
    uint32_t* d_hist = nullptr;
    uint32_t* d_hist_T = nullptr;

    try {
        const size_t num_blocks = 3;
        const size_t num_bins = 4;

        // Input: rows are blocks.
        const std::vector<uint32_t> hist = {
            2, 1, 1, 0,
            1, 2, 1, 0,
            0, 1, 1, 2
        };

        // Expected transpose: rows are bins.
        const std::vector<uint32_t> expected = {
            2, 1, 0,
            1, 2, 1,
            1, 1, 1,
            0, 0, 2
        };

        const size_t bytes = hist.size() * sizeof(uint32_t);
        std::vector<uint32_t> actual(hist.size());

        // Allocate input and output arrays on the GPU.
        check_cuda(cudaMalloc(
            reinterpret_cast<void**>(&d_hist), bytes
        ));
        check_cuda(cudaMalloc(
            reinterpret_cast<void**>(&d_hist_T), bytes
        ));

        // Copy the histogram from CPU memory to GPU memory.
        check_cuda(cudaMemcpy(
            d_hist,
            hist.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        // Launch the transpose kernel.
        const unsigned threads_per_block = 256;
        const unsigned launch_blocks = 1;

        transpose_histogram<<<launch_blocks, threads_per_block>>>(
            d_hist, d_hist_T, num_blocks, num_bins
        );

        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());

        // Copy the transposed histogram back to CPU memory.
        check_cuda(cudaMemcpy(
            actual.data(),
            d_hist_T,
            bytes,
            cudaMemcpyDeviceToHost
        ));

        std::cout << "Transposed histogram (rows are bins):\n";

        for (size_t bin = 0; bin < num_bins; ++bin) {
            std::cout << "Bin " << bin << ": ";

            for (size_t block = 0; block < num_blocks; ++block) {
                std::cout << actual[bin * num_blocks + block] << " ";
            }

            std::cout << "\n";
        }

        // Release GPU memory.
        check_cuda(cudaFree(d_hist));
        d_hist = nullptr;

        check_cuda(cudaFree(d_hist_T));
        d_hist_T = nullptr;

        if (actual != expected) {
            std::cerr << "FAIL: Transpose differs from expected values.\n";
            return 1;
        }

        std::cout << "PASS: All 12 transposed counts match.\n";
        return 0;

    } catch (const std::exception& error) {
        if (d_hist != nullptr) {
            cudaFree(d_hist);
        }

        if (d_hist_T != nullptr) {
            cudaFree(d_hist_T);
        }

        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}