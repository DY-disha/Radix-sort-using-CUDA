
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
// CUDA error checking
// ============================================================

void check_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

// ============================================================
// GPU memory management
// ============================================================

struct DeviceBuffer {
    uint32_t* data = nullptr;

    explicit DeviceBuffer(size_t n) {
        if (n > 0) {
            check_cuda(cudaMalloc(
                reinterpret_cast<void**>(&data),
                n * sizeof(uint32_t)
            ));
        }
    }

    ~DeviceBuffer() {
        if (data != nullptr) {
            cudaFree(data);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// ============================================================
// CUDA event management
// ============================================================

struct CudaEvent {
    cudaEvent_t event = nullptr;

    CudaEvent() {
        check_cuda(cudaEventCreate(&event));
    }

    ~CudaEvent() {
        if (event != nullptr) {
            cudaEventDestroy(event);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
};

// ============================================================
// CUDA stream management
// ============================================================

struct CudaStream {
    cudaStream_t stream = nullptr;

    CudaStream() {
        check_cuda(cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking
        ));
    }

    ~CudaStream() {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
};

// ============================================================
// Input generation
// ============================================================
//
// All arrays contain unsigned 32-bit integers.
//
// Patterns:
// random
// all_equal
// few_distinct
// sorted
// reversed
//
// A fixed seed makes the random inputs reproducible.
// ============================================================

std::vector<uint32_t> generate_input(
    size_t n,
    const std::string& pattern
) {
    std::vector<uint32_t> input(n);

    std::mt19937 rng(2026);

    for (size_t i = 0; i < n; ++i) {
        if (pattern == "random") {
            input[i] = rng();
        }
        else if (pattern == "all_equal") {
            input[i] = UINT32_MAX;
        }
        else if (pattern == "few_distinct") {
            input[i] = static_cast<uint32_t>(i % 7);
        }
        else if (pattern == "sorted") {
            input[i] = static_cast<uint32_t>(i);
        }
        else if (pattern == "reversed") {
            input[i] = static_cast<uint32_t>(n - i);
        }
        else {
            throw std::invalid_argument(
                "Unknown input pattern"
            );
        }
    }

    return input;
}

// ============================================================
// Median calculation
// ============================================================

double calculate_median(
    std::vector<double> times
) {
    std::sort(times.begin(), times.end());

    const size_t n = times.size();

    if (n == 0) {
        throw std::invalid_argument(
            "No timing measurements"
        );
    }

    if (n % 2 == 1) {
        return times[n / 2];
    }

    return (times[n / 2 - 1] + times[n / 2]) / 2.0;
}

// ============================================================
// CUB full-sort benchmark
// ============================================================

double benchmark_cub_sort(
    const std::vector<uint32_t>& input,
    int warmup_runs,
    int measured_runs
) {
    const size_t n = input.size();

    if (n == 0 || warmup_runs < 0 || measured_runs <= 0) {
        throw std::invalid_argument(
            "Invalid benchmark parameters"
        );
    }

    if (n > static_cast<size_t>(INT32_MAX)) {
        throw std::invalid_argument(
            "Input exceeds supported benchmark size"
        );
    }

    const size_t bytes = n * sizeof(uint32_t);

    CudaStream stream;

    DeviceBuffer d_original(n);
    DeviceBuffer d_input(n);
    DeviceBuffer d_output(n);

    CudaEvent start;
    CudaEvent stop;

    // --------------------------------------------------------
    // Copy original unsorted input to GPU once.
    // --------------------------------------------------------

    check_cuda(cudaMemcpy(
        d_original.data,
        input.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    // --------------------------------------------------------
    // Query CUB temporary storage.
    // --------------------------------------------------------

    size_t temp_storage_bytes = 0;

    check_cuda(cub::DeviceRadixSort::SortKeys(
        nullptr,
        temp_storage_bytes,
        d_input.data,
        d_output.data,
        static_cast<int>(n),
        0,
        32,
        stream.stream
    ));

    void* d_temp_storage = nullptr;

    check_cuda(cudaMalloc(
        &d_temp_storage,
        temp_storage_bytes
    ));

    try {
        // ----------------------------------------------------
        // Restore original unsorted input.
        //
        // This operation is outside the measured sorting time.
        // ----------------------------------------------------

        auto restore_input = [&]() {
            check_cuda(cudaMemcpyAsync(
                d_input.data,
                d_original.data,
                bytes,
                cudaMemcpyDeviceToDevice,
                stream.stream
            ));
        };

        // ----------------------------------------------------
        // Run one complete 32-bit CUB radix sort.
        // ----------------------------------------------------

        auto run_sort = [&]() {
            check_cuda(cub::DeviceRadixSort::SortKeys(
                d_temp_storage,
                temp_storage_bytes,
                d_input.data,
                d_output.data,
                static_cast<int>(n),
                0,
                32,
                stream.stream
            ));
        };

        // ----------------------------------------------------
        // Warm-up runs
        // ----------------------------------------------------

        for (int i = 0; i < warmup_runs; ++i) {
            restore_input();
            run_sort();
        }

        check_cuda(cudaStreamSynchronize(stream.stream));

        // ----------------------------------------------------
        // Measured runs
        //
        // The timer includes only the complete CUB sort.
        // Input restoration is excluded.
        // ----------------------------------------------------

        std::vector<double> times;
        times.reserve(measured_runs);

        for (int i = 0; i < measured_runs; ++i) {
            restore_input();

            check_cuda(cudaEventRecord(
                start.event,
                stream.stream
            ));

            run_sort();

            check_cuda(cudaEventRecord(
                stop.event,
                stream.stream
            ));

            check_cuda(cudaEventSynchronize(stop.event));

            float elapsed_ms = 0.0f;

            check_cuda(cudaEventElapsedTime(
                &elapsed_ms,
                start.event,
                stop.event
            ));

            times.push_back(
                static_cast<double>(elapsed_ms)
            );
        }

        // ----------------------------------------------------
        // Correctness verification
        //
        // Compare CUB output against CPU std::sort.
        // Outside the timed region.
        // ----------------------------------------------------

        std::vector<uint32_t> expected = input;
        std::sort(expected.begin(), expected.end());

        std::vector<uint32_t> actual(n);

        check_cuda(cudaMemcpy(
            actual.data(),
            d_output.data,
            bytes,
            cudaMemcpyDeviceToHost
        ));

        if (actual != expected) {
            size_t index = 0;

            while (index < n &&
                   actual[index] == expected[index]) {
                ++index;
            }

            throw std::runtime_error(
                "CUB sorting mismatch at index "
                + std::to_string(index)
            );
        }

        const double median_ms =
            calculate_median(times);

        check_cuda(cudaFree(d_temp_storage));
        d_temp_storage = nullptr;

        return median_ms;

    } catch (...) {
        if (d_temp_storage != nullptr) {
            cudaFree(d_temp_storage);
        }

        throw;
    }
}

// ============================================================
// Main benchmark
// ============================================================

int main() {
    try {
        const int warmup_runs = 5;
        const int measured_runs = 20;

        const std::vector<size_t> sizes = {
            10000,
            100000,
            1000000
        };

        const std::vector<std::string> patterns = {
            "random",
            "all_equal",
            "few_distinct",
            "sorted",
            "reversed"
        };

        // Print GPU information to stderr so CSV output
        // remains easy to save and compare.

        int device = 0;
        check_cuda(cudaGetDevice(&device));

        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(
            &properties,
            device
        ));

        std::cerr << "GPU: " << properties.name << "\n";
        std::cerr << "Warm-up runs: " << warmup_runs << "\n";
        std::cerr << "Measured runs: " << measured_runs << "\n";

        std::cout << std::fixed
                  << std::setprecision(6);

        std::cout
            << "implementation,n,pattern,median_ms,status\n";

        size_t passed = 0;

        for (size_t n : sizes) {
            for (const auto& pattern : patterns) {
                const auto input =
                    generate_input(n, pattern);

                const double median_ms =
                    benchmark_cub_sort(
                        input,
                        warmup_runs,
                        measured_runs
                    );

                std::cout
                    << "CUB,"
                    << n << ","
                    << pattern << ","
                    << median_ms << ","
                    << "PASS\n";

                ++passed;
            }
        }

        std::cerr
            << "PASS: All "
            << passed
            << " CUB full-sort benchmark cases "
            << "matched CPU std::sort.\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "ERROR: "
            << error.what()
            << "\n";

        return 1;
    }
}
