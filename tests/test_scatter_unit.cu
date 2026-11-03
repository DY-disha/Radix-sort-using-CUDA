#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "radix_config.h"
#include "reference_offsets.h"
#include "scatter.cuh"

namespace {

void check_cuda(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(result));
    }
}

std::vector<uint32_t> expected_scatter(
    const std::vector<uint32_t>& input,
    int shift)
{
    std::vector<uint32_t> output;
    output.reserve(input.size());

    for (unsigned digit = 0; digit < OFFSET_NUM_BINS; ++digit)
    {
        for (uint32_t value : input)
        {
            if (((value >> shift) & (OFFSET_NUM_BINS - 1)) == digit)
                output.push_back(value);
        }
    }

    return output;
}

std::vector<uint32_t> make_input(size_t n, unsigned pattern)
{
    std::vector<uint32_t> input(n);
    std::mt19937 rng(0x5eed1234u + static_cast<unsigned>(n) + pattern);

    for (size_t index = 0; index < n; ++index)
    {
        switch (pattern)
        {
        case 0:
            input[index] = rng();
            break;
        case 1:
            // Exercise stability: many equal digits with distinct values.
            input[index] =
                static_cast<uint32_t>((index << 8) | (index % 3));
            break;
        case 2:
            // Exercise every possible digit, including partial final tiles.
            input[index] =
                static_cast<uint32_t>((index * 37u) & 0xffu) << 16 |
                static_cast<uint32_t>(index & 0xffffu);
            break;
        default:
            input[index] =
                static_cast<uint32_t>((n - index) * 2654435761u);
            break;
        }
    }

    return input;
}

void run_case(size_t n, unsigned pattern, int shift)
{
    const std::vector<uint32_t> input = make_input(n, pattern);
    const std::vector<uint32_t> expected =
        expected_scatter(input, shift);

    int num_blocks = 0;
    std::vector<uint32_t> offsets;
    generate_reference_offsets(input, shift, num_blocks, offsets);

    uint32_t* device_input = nullptr;
    uint32_t* device_output = nullptr;
    uint32_t* device_offsets = nullptr;
    cudaStream_t stream = nullptr;

    try
    {
        check_cuda(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
        check_cuda(
            cudaMalloc(&device_input, n * sizeof(uint32_t)),
            "cudaMalloc(input)");
        check_cuda(
            cudaMalloc(&device_output, n * sizeof(uint32_t)),
            "cudaMalloc(output)");
        check_cuda(
            cudaMalloc(
                &device_offsets,
                offsets.size() * sizeof(uint32_t)),
            "cudaMalloc(offsets)");

        check_cuda(
            cudaMemcpyAsync(
                device_input,
                input.data(),
                n * sizeof(uint32_t),
                cudaMemcpyHostToDevice,
                stream),
            "cudaMemcpyAsync(input)");
        check_cuda(
            cudaMemcpyAsync(
                device_offsets,
                offsets.data(),
                offsets.size() * sizeof(uint32_t),
                cudaMemcpyHostToDevice,
                stream),
            "cudaMemcpyAsync(offsets)");

        check_cuda(
            launch_stable_scatter(
                device_input,
                device_output,
                device_offsets,
                n,
                shift,
                stream),
            "launch_stable_scatter");
        check_cuda(
            cudaGetLastError(),
            "launch_stable_scatter launch");

        std::vector<uint32_t> actual(n);
        check_cuda(
            cudaMemcpyAsync(
                actual.data(),
                device_output,
                n * sizeof(uint32_t),
                cudaMemcpyDeviceToHost,
                stream),
            "cudaMemcpyAsync(output)");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        if (actual != expected)
        {
            throw std::runtime_error(
                "output mismatch for N=" + std::to_string(n) +
                ", pattern=" + std::to_string(pattern) +
                ", shift=" + std::to_string(shift));
        }
    }
    catch (...)
    {
        if (device_offsets != nullptr)
            cudaFree(device_offsets);
        if (device_output != nullptr)
            cudaFree(device_output);
        if (device_input != nullptr)
            cudaFree(device_input);
        if (stream != nullptr)
            cudaStreamDestroy(stream);
        throw;
    }

    check_cuda(cudaFree(device_offsets), "cudaFree(offsets)");
    check_cuda(cudaFree(device_output), "cudaFree(output)");
    check_cuda(cudaFree(device_input), "cudaFree(input)");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

} // namespace

int main()
{
    try
    {
        const size_t tile_size =
            static_cast<size_t>(OFFSET_ITEMS_PER_BLOCK);
        const std::vector<size_t> sizes = {
            1,
            31,
            32,
            33,
            255,
            256,
            257,
            tile_size - 1,
            tile_size,
            tile_size + 1,
            2 * tile_size + 17
        };

        size_t cases = 0;
        for (size_t n : sizes)
        {
            for (unsigned pattern = 0; pattern < 4; ++pattern)
            {
                for (int shift : {0, 8, 16, 24})
                {
                    run_case(n, pattern, shift);
                    ++cases;
                }
            }
        }

        std::cout
            << "SCATTER UNIT TESTS PASSED: "
            << cases << " cases.\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Scatter unit test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
