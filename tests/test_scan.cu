#include "global_offsets/scan.cuh"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

void check_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

// Calculate expected results independently on the CPU.
std::vector<uint32_t> scan_cpu(
    const std::vector<uint32_t>& input
) {
    std::vector<uint32_t> output(input.size());

    uint64_t sum = 0;

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<uint32_t>(sum);

        sum += input[i];

        if (sum > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error(
                "Input sum exceeds uint32_t"
            );
        }
    }

    return output;
}

void compare_result(
    const std::vector<uint32_t>& actual,
    const std::vector<uint32_t>& expected,
    const std::string& name,
    const std::string& implementation
) {
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(
                name
                + " [" + implementation + "]"
                + ": mismatch at index "
                + std::to_string(i)
                + ", expected "
                + std::to_string(expected[i])
                + ", got "
                + std::to_string(actual[i])
            );
        }
    }
}

void test_scan(
    const std::vector<uint32_t>& input,
    const std::vector<uint32_t>& expected,
    const std::string& name
) {
    uint32_t* d_input = nullptr;
    uint32_t* d_output_original = nullptr;
    uint32_t* d_output_workspace = nullptr;
    uint32_t* d_workspace = nullptr;

    try {
        if (input.size() != expected.size()) {
            throw std::invalid_argument(
                "Test sizes do not match"
            );
        }

        // ----------------------------------------------------
        // Empty input
        // ----------------------------------------------------
        if (input.empty()) {
            check_cuda(
                offset_scan::exclusive_scan(
                    nullptr,
                    nullptr,
                    0
                )
            );

            check_cuda(
                offset_scan::exclusive_scan_with_workspace(
                    nullptr,
                    nullptr,
                    0,
                    nullptr,
                    0
                )
            );

            std::cout
                << "PASS: "
                << name
                << " [original + workspace]\n";

            return;
        }

        const size_t n = input.size();

        const size_t bytes =
            n * sizeof(uint32_t);

        // ----------------------------------------------------
        // Allocate input and outputs
        // ----------------------------------------------------

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_input),
                bytes
            )
        );

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_output_original),
                bytes
            )
        );

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&d_output_workspace),
                bytes
            )
        );

        check_cuda(
            cudaMemcpy(
                d_input,
                input.data(),
                bytes,
                cudaMemcpyHostToDevice
            )
        );

        // ----------------------------------------------------
        // Original implementation
        // ----------------------------------------------------

        check_cuda(
            offset_scan::exclusive_scan(
                d_input,
                d_output_original,
                n
            )
        );

        // ----------------------------------------------------
        // Workspace implementation
        // ----------------------------------------------------

        const size_t workspace_elements =
            offset_scan::workspace_elements_required(n);

        if (workspace_elements > 0) {
            check_cuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&d_workspace),
                    workspace_elements * sizeof(uint32_t)
                )
            );
        }

        check_cuda(
            offset_scan::exclusive_scan_with_workspace(
                d_input,
                d_output_workspace,
                n,
                d_workspace,
                workspace_elements
            )
        );

        check_cuda(cudaDeviceSynchronize());

        // ----------------------------------------------------
        // Copy results back
        // ----------------------------------------------------

        std::vector<uint32_t> actual_original(n);
        std::vector<uint32_t> actual_workspace(n);

        check_cuda(
            cudaMemcpy(
                actual_original.data(),
                d_output_original,
                bytes,
                cudaMemcpyDeviceToHost
            )
        );

        check_cuda(
            cudaMemcpy(
                actual_workspace.data(),
                d_output_workspace,
                bytes,
                cudaMemcpyDeviceToHost
            )
        );

        // ----------------------------------------------------
        // Compare both implementations with CPU reference
        // ----------------------------------------------------

        compare_result(
            actual_original,
            expected,
            name,
            "original"
        );

        compare_result(
            actual_workspace,
            expected,
            name,
            "workspace"
        );

        // Also verify that both GPU implementations agree.
        for (size_t i = 0; i < n; ++i) {
            if (actual_original[i] != actual_workspace[i]) {
                throw std::runtime_error(
                    name
                    + ": GPU implementations differ at index "
                    + std::to_string(i)
                );
            }
        }

        std::cout
            << "PASS: "
            << name
            << " [original + workspace]\n";

        if (name == "12-entry histogram") {
            std::cout << "Offsets: ";

            for (uint32_t value : actual_workspace) {
                std::cout << value << " ";
            }

            std::cout << "\n";
        }

        // ----------------------------------------------------
        // Cleanup
        // ----------------------------------------------------

        if (d_workspace != nullptr) {
            check_cuda(cudaFree(d_workspace));
            d_workspace = nullptr;
        }

        check_cuda(cudaFree(d_input));
        d_input = nullptr;

        check_cuda(cudaFree(d_output_original));
        d_output_original = nullptr;

        check_cuda(cudaFree(d_output_workspace));
        d_output_workspace = nullptr;

    } catch (...) {
        if (d_workspace != nullptr) {
            cudaFree(d_workspace);
        }

        if (d_input != nullptr) {
            cudaFree(d_input);
        }

        if (d_output_original != nullptr) {
            cudaFree(d_output_original);
        }

        if (d_output_workspace != nullptr) {
            cudaFree(d_output_workspace);
        }

        throw;
    }
}

int main() {
    try {
        // ----------------------------------------------------
        // Hand-calculated example
        // ----------------------------------------------------

        test_scan(
            {
                2, 1, 0, 1,
                2, 1, 1, 1,
                1, 0, 0, 2
            },
            {
                0, 2, 3, 3,
                4, 6, 7, 8,
                9, 10, 10, 10
            },
            "12-entry histogram"
        );

        // ----------------------------------------------------
        // Boundary and larger cases
        // ----------------------------------------------------

        const std::vector<size_t> sizes = {
            0,
            1,
            2,
            3,
            255,
            256,
            257,
            511,
            512,
            513,
            514,
            1000,
            1023,
            1024,
            1025,
            2048,
            4096,
            10000
        };

        size_t passed = 1;

        for (size_t n : sizes) {
            for (unsigned mode = 0;
                 mode < 3;
                 ++mode) {

                std::vector<uint32_t> input(n);

                for (size_t i = 0;
                     i < n;
                     ++i) {

                    if (mode == 0) {
                        // All zeros.
                        input[i] = 0;

                    } else if (mode == 1) {
                        // All ones.
                        input[i] = 1;

                    } else {
                        // Deterministic non-uniform data.
                        input[i] =
                            static_cast<uint32_t>(
                                (i * 7 + 3) % 11
                            );
                    }
                }

                test_scan(
                    input,
                    scan_cpu(input),
                    "n="
                    + std::to_string(n)
                    + ", mode="
                    + std::to_string(mode)
                );

                ++passed;
            }
        }

        std::cout
            << "PASS: All "
            << passed
            << " scan cases matched expected results "
            << "for both implementations.\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr
            << "ERROR: "
            << error.what()
            << "\n";

        return 1;
    }
}