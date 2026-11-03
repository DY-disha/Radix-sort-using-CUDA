#include "radix_config.h"
#include "reference_offsets.h"
#include "scatter.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

struct DeviceBuffer {
    uint32_t* data = nullptr;

    explicit DeviceBuffer(size_t count)
    {
        if (count == 0)
            throw std::invalid_argument("DeviceBuffer requires a nonzero count");
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(uint32_t)),
            "cudaMalloc");
    }

    ~DeviceBuffer()
    {
        if (data != nullptr)
            cudaFree(data);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Stream {
    cudaStream_t value = nullptr;

    Stream()
    {
        check_cuda(
            cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }

    ~Stream()
    {
        if (value != nullptr)
            cudaStreamDestroy(value);
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

struct Event {
    cudaEvent_t value = nullptr;

    Event()
    {
        check_cuda(cudaEventCreate(&value), "cudaEventCreate");
    }

    ~Event()
    {
        if (value != nullptr)
            cudaEventDestroy(value);
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

std::vector<uint32_t> generate_input(size_t n, const std::string& pattern)
{
    std::vector<uint32_t> input(n);
    std::mt19937 rng(2026);

    for (size_t index = 0; index < n; ++index)
    {
        if (pattern == "random")
            input[index] = rng();
        else if (pattern == "all_equal")
            input[index] = UINT32_MAX;
        else if (pattern == "few_distinct")
            input[index] = static_cast<uint32_t>(index % 7);
        else if (pattern == "sorted")
            input[index] = static_cast<uint32_t>(index);
        else if (pattern == "reversed")
            input[index] = static_cast<uint32_t>(n - index);
        else
            throw std::invalid_argument("Unknown pattern: " + pattern);
    }

    return input;
}

std::vector<uint32_t> expected_scatter(
    const std::vector<uint32_t>& input,
    int shift)
{
    std::vector<uint32_t> expected;
    expected.reserve(input.size());

    for (unsigned digit = 0; digit < radix_config::NUM_BINS; ++digit)
    {
        for (uint32_t value : input)
        {
            if (((value >> shift) & (radix_config::NUM_BINS - 1)) == digit)
                expected.push_back(value);
        }
    }

    return expected;
}

void validate_case(
    const std::vector<uint32_t>& input,
    int shift,
    DeviceBuffer& device_input,
    DeviceBuffer& device_output,
    DeviceBuffer& device_offsets,
    Stream& stream)
{
    const std::vector<uint32_t> expected = expected_scatter(input, shift);
    std::vector<uint32_t> actual(input.size());

    check_cuda(
        cudaMemsetAsync(
            device_output.data,
            0xa5,
            input.size() * sizeof(uint32_t),
            stream.value),
        "cudaMemsetAsync(output)");
    check_cuda(
        launch_stable_scatter(
            device_input.data,
            device_output.data,
            device_offsets.data,
            input.size(),
            shift,
            stream.value),
        "launch_stable_scatter");
    check_cuda(
        cudaMemcpyAsync(
            actual.data(),
            device_output.data,
            input.size() * sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream.value),
        "cudaMemcpyAsync(output)");
    check_cuda(cudaStreamSynchronize(stream.value), "cudaStreamSynchronize");

    if (actual != expected)
    {
        throw std::runtime_error(
            "scatter mismatch for n=" + std::to_string(input.size()) +
            ", shift=" + std::to_string(shift));
    }
}

void upload_case(
    const std::vector<uint32_t>& input,
    const std::vector<uint32_t>& offsets,
    DeviceBuffer& device_input,
    DeviceBuffer& device_offsets,
    Stream& stream)
{
    check_cuda(
        cudaMemcpyAsync(
            device_input.data,
            input.data(),
            input.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            stream.value),
        "cudaMemcpyAsync(input)");
    check_cuda(
        cudaMemcpyAsync(
            device_offsets.data,
            offsets.data(),
            offsets.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            stream.value),
        "cudaMemcpyAsync(offsets)");
    check_cuda(cudaStreamSynchronize(stream.value), "cudaStreamSynchronize(upload)");
}

struct Timing {
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

Timing measure(
    const std::vector<uint32_t>& input,
    int shift,
    int batch_runs,
    int warmup_runs,
    int measured_samples,
    DeviceBuffer& device_input,
    DeviceBuffer& device_output,
    DeviceBuffer& device_offsets,
    Stream& stream,
    Event& start,
    Event& stop)
{
    for (int run = 0; run < warmup_runs * batch_runs; ++run)
    {
        check_cuda(
            launch_stable_scatter(
                device_input.data,
                device_output.data,
                device_offsets.data,
                input.size(),
                shift,
                stream.value),
            "launch_stable_scatter(warmup)");
    }
    check_cuda(cudaStreamSynchronize(stream.value), "cudaStreamSynchronize(warmup)");

    std::vector<double> samples;
    samples.reserve(measured_samples);
    for (int sample = 0; sample < measured_samples; ++sample)
    {
        check_cuda(cudaEventRecord(start.value, stream.value), "cudaEventRecord(start)");
        for (int run = 0; run < batch_runs; ++run)
        {
            check_cuda(
                launch_stable_scatter(
                    device_input.data,
                    device_output.data,
                    device_offsets.data,
                    input.size(),
                    shift,
                    stream.value),
                "launch_stable_scatter(measurement)");
        }
        check_cuda(cudaEventRecord(stop.value, stream.value), "cudaEventRecord(stop)");
        check_cuda(cudaEventSynchronize(stop.value), "cudaEventSynchronize");

        float elapsed_ms = 0.0f;
        check_cuda(
            cudaEventElapsedTime(&elapsed_ms, start.value, stop.value),
            "cudaEventElapsedTime");
        samples.push_back(static_cast<double>(elapsed_ms) / batch_runs);
    }

    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    const double median_ms = samples.size() % 2 == 0
        ? (samples[middle - 1] + samples[middle]) / 2.0
        : samples[middle];
    return {median_ms, samples.front(), samples.back()};
}

void run_boundary_checks()
{
    const size_t tile = radix_config::TILE;
    const std::vector<size_t> sizes = {
        1, 31, 32, 33, 255, 256, 257,
        tile - 1, tile, tile + 1, 2 * tile + 17
    };
    size_t checks = 0;

    for (size_t n : sizes)
    {
        for (const std::string& pattern : {"random", "all_equal", "few_distinct"})
        {
            const std::vector<uint32_t> input = generate_input(n, pattern);
            for (int shift : {0, 8, 16, 24})
            {
                int num_blocks = 0;
                std::vector<uint32_t> offsets;
                generate_reference_offsets(input, shift, num_blocks, offsets);
                Stream stream;
                DeviceBuffer device_input(n);
                DeviceBuffer device_output(n);
                DeviceBuffer device_offsets(offsets.size());
                upload_case(input, offsets, device_input, device_offsets, stream);
                validate_case(
                    input,
                    shift,
                    device_input,
                    device_output,
                    device_offsets,
                    stream);
                ++checks;
            }
        }
    }

    std::cerr << "PASS: " << checks << " scatter boundary/reference checks.\n";
}

void run_benchmark_case(
    size_t n,
    const std::string& pattern,
    int shift,
    int batch_runs,
    int warmup_runs,
    int measured_samples)
{
    const std::vector<uint32_t> input = generate_input(n, pattern);
    int num_blocks = 0;
    std::vector<uint32_t> offsets;
    generate_reference_offsets(input, shift, num_blocks, offsets);

    Stream stream;
    DeviceBuffer device_input(n);
    DeviceBuffer device_output(n);
    DeviceBuffer device_offsets(offsets.size());
    Event start;
    Event stop;
    upload_case(input, offsets, device_input, device_offsets, stream);

    const Timing timing = measure(
        input,
        shift,
        batch_runs,
        warmup_runs,
        measured_samples,
        device_input,
        device_output,
        device_offsets,
        stream,
        start,
        stop);
    validate_case(
        input,
        shift,
        device_input,
        device_output,
        device_offsets,
        stream);

    std::cout << n << ',' << pattern << ',' << shift << ','
              << radix_config::B << ',' << radix_config::Q << ','
              << radix_config::RADIX_BITS << ',' << radix_config::NUM_BINS << ','
              << num_blocks << ','
              << timing.median_ms << ',' << timing.min_ms << ','
              << timing.max_ms << ",PASS\n";
}

} // namespace

int main()
{
    try
    {
        constexpr int batch_runs = 100;
        constexpr int warmup_runs = 5;
        constexpr int measured_samples = 20;

        int device = 0;
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        check_cuda(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties");

        std::cerr << "GPU: " << properties.name << '\n'
                  << "CUDA configuration: B=" << radix_config::B
                  << " Q=" << radix_config::Q
                  << " r=" << radix_config::RADIX_BITS
                  << " bins=" << radix_config::NUM_BINS
                  << " tile=" << radix_config::TILE << '\n'
                  << "Warm-up batches: " << warmup_runs << '\n'
                  << "Measured samples: " << measured_samples << '\n'
                  << "Calls per sample: " << batch_runs << '\n'
                  << "Timing excludes allocation, offset generation, transfers, and validation.\n";

        run_boundary_checks();

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "n,pattern,shift,B,Q,r,bins,blocks,median_ms,min_ms,max_ms,status\n";

        size_t cases = 0;
        for (size_t n : {size_t(10000), size_t(100000), size_t(1000000)})
        {
            for (const std::string& pattern : {
                "random", "all_equal", "few_distinct", "sorted", "reversed"
            })
            {
                for (int shift : {0, 8, 16, 24})
                {
                    run_benchmark_case(
                        n,
                        pattern,
                        shift,
                        batch_runs,
                        warmup_runs,
                        measured_samples);
                    ++cases;
                }
            }
        }

        std::cerr << "PASS: " << cases
                  << " scatter benchmark cases matched the CPU reference.\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
