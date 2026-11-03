
#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "radix_config.h"
#include "radix_sort_cpu.h"
#include "reference_offsets.h"
#include "scatter.cuh"

uint32_t parse_value(const std::string& text)
{
    size_t parsed = 0;
    const unsigned long long value = std::stoull(text, &parsed, 0);

    if (parsed != text.size() ||
        value > std::numeric_limits<uint32_t>::max())
    {
        throw std::invalid_argument(
            "invalid uint32 value: " + text);
    }

    return static_cast<uint32_t>(value);
}

std::vector<uint32_t> read_input(int argc, char** argv)
{
    std::vector<uint32_t> input;
    const int first_value =
        (argc > 1 && std::string(argv[1]) == "--passes") ? 2 : 1;

    if (argc == 1)
    {
        return {
            0x00010000,
            0x00000002,
            0x00000100,
            0x00000001
        };
    }

    if (argc == first_value + 2 &&
        std::string(argv[first_value]) == "--file")
    {
        std::ifstream file(argv[first_value + 1]);
        if (!file)
            throw std::runtime_error(
                "could not open input file: " +
                std::string(argv[first_value + 1]));

        std::string value;
        while (file >> value)
            input.push_back(parse_value(value));
    }
    else if (first_value < argc &&
             std::string(argv[first_value]) != "--file")
    {
        for (int index = first_value; index < argc; ++index)
            input.push_back(parse_value(argv[index]));
    }
    else
    {
        throw std::invalid_argument(
            "usage: test_scatter [VALUE ...] or test_scatter --file FILE");
    }

    if (input.empty())
        throw std::invalid_argument("input must contain at least one value");

    return input;
}

void print_usage(const char* program)
{
    std::cout << "Usage:\n"
              << "  " << program << "\n"
              << "  " << program << " VALUE [VALUE ...]\n"
              << "  " << program << " --file FILE\n"
              << "  " << program << " --passes VALUE [VALUE ...]\n"
              << "  " << program << " --passes --file FILE\n";
}

void print_output(
    const std::vector<uint32_t>& cpu_output,
    const std::vector<uint32_t>& gpu_output)
{
    for (size_t index = 0; index < cpu_output.size(); ++index)
    {
        std::cout << "0x" << std::hex << cpu_output[index]
                  << " 0x" << gpu_output[index] << std::dec << '\n';
    }
}

void print_pass_output(
    const std::vector<uint32_t>& pass_input,
    const std::vector<uint32_t>& cpu_output,
    const std::vector<uint32_t>& gpu_output)
{
    for (size_t index = 0; index < pass_input.size(); ++index)
    {
        std::cout << "0x" << std::hex << pass_input[index]
                  << " 0x" << cpu_output[index]
                  << " 0x" << gpu_output[index] << std::dec << '\n';
    }
}

int main(int argc, char** argv)
{
    const bool print_passes =
        argc > 1 && std::string(argv[1]) == "--passes";

    if (argc == 2 &&
        (std::string(argv[1]) == "--help" || print_passes))
    {
        if (std::string(argv[1]) == "--help")
        {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::vector<uint32_t> input;
    try
    {
        input = read_input(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }

    const unsigned num_passes =
        (32 + OFFSET_RADIX_BITS - 1) / OFFSET_RADIX_BITS;

    std::vector<uint32_t> cpu_output = input;
    std::vector<std::vector<uint32_t>> cpu_pass_outputs;
    for (unsigned pass = 0; pass < num_passes; ++pass)
    {
        cpu_output = radixSortPassCPU(
            cpu_output,
            static_cast<unsigned>(OFFSET_RADIX_BITS),
            pass);

        if (print_passes)
            cpu_pass_outputs.push_back(cpu_output);
    }

    const size_t n = input.size();
    std::vector<uint32_t> gpu_input = input;
    std::vector<std::vector<uint32_t>> gpu_pass_inputs;
    uint32_t* d_input = nullptr;
    uint32_t* d_output = nullptr;
    uint32_t* d_offsets = nullptr;

    cudaMalloc(&d_input, n * sizeof(uint32_t));
    cudaMalloc(&d_output, n * sizeof(uint32_t));

    cudaMemcpy(
        d_input, input.data(),
        n * sizeof(uint32_t),
        cudaMemcpyHostToDevice);

    for (unsigned pass = 0; pass < num_passes; ++pass)
    {
        const int shift =
            static_cast<int>(pass * OFFSET_RADIX_BITS);
        if (print_passes)
            gpu_pass_inputs.push_back(gpu_input);

        int num_blocks = 0;
        std::vector<uint32_t> offsets;

        generate_reference_offsets(
            gpu_input, shift, num_blocks, offsets);

        cudaMalloc(
            &d_offsets,
            offsets.size() * sizeof(uint32_t));

        cudaMemcpy(
            d_offsets, offsets.data(),
            offsets.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice);

        cudaError_t err = launch_stable_scatter(
            d_input, d_output, d_offsets, n, shift);

        if (err != cudaSuccess)
        {
            std::cerr << "Scatter launch failed on pass " << pass << ": "
                      << cudaGetErrorString(err) << '\n';
            return 1;
        }

        err = cudaDeviceSynchronize();

        if (err != cudaSuccess)
        {
            std::cerr << "Scatter execution failed on pass " << pass << ": "
                      << cudaGetErrorString(err) << '\n';
            return 1;
        }

        cudaMemcpy(
            gpu_input.data(), d_output,
            n * sizeof(uint32_t),
            cudaMemcpyDeviceToHost);

        cudaFree(d_offsets);
        d_offsets = nullptr;

        uint32_t* temporary = d_input;
        d_input = d_output;
        d_output = temporary;

        if (print_passes)
        {
            std::cout << "Pass " << pass << ":\n";
            print_pass_output(
                gpu_pass_inputs[pass],
                cpu_pass_outputs[pass],
                gpu_input);
        }
    }

    std::vector<uint32_t> gpu_output(n);

    cudaMemcpy(
        gpu_output.data(), d_input,
        n * sizeof(uint32_t),
        cudaMemcpyDeviceToHost);

    if (!print_passes)
        print_output(cpu_output, gpu_output);

    const bool matches = cpu_output == gpu_output;
    std::cout << "Outputs match: " << std::boolalpha << matches << '\n';

    cudaFree(d_input);
    cudaFree(d_output);

    return matches ? 0 : 1;
}
