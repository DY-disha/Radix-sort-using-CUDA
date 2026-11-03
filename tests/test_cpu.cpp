#include "offset_reference.hpp"

#include <iostream>
#include <vector>

int main() {
    try {
        const size_t num_blocks = 3;
        const size_t num_bins = 4;

        // Input: one row per block.
        const std::vector<uint32_t> hist = {
            2, 1, 1, 0,
            1, 2, 1, 0,
            0, 1, 1, 2
        };

        // Hand-calculated offsets: one row per bin.
        const std::vector<uint32_t> expected = {
            0, 2, 3,
            3, 4, 6,
            7, 8, 9,
            10, 10, 10
        };

        const auto actual =
            compute_offsets_cpu(hist, num_blocks, num_bins);

        std::cout << "Calculated offsets (rows are bins):\n";

        for (size_t bin = 0; bin < num_bins; ++bin) {
            std::cout << "Bin " << bin << ": ";

            for (size_t block = 0; block < num_blocks; ++block) {
                std::cout << actual[bin * num_blocks + block] << " ";
            }

            std::cout << "\n";
        }

        if (actual != expected) {
            std::cerr << "FAIL: Offsets differ from expected values.\n";
            return 1;
        }

        std::cout << "PASS: All 12 offsets match the expected values.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}