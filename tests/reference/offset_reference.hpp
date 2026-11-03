#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

inline std::vector<uint32_t> compute_offsets_cpu(
    const std::vector<uint32_t>& hist,
    size_t num_blocks,
    size_t num_bins
) {
    // Check that the dimensions match the input array.
    if (num_bins == 0) {
        throw std::invalid_argument("num_bins must be positive");
    }

    if (hist.size() % num_bins != 0 ||
        hist.size() / num_bins != num_blocks) {
        throw std::invalid_argument("Histogram dimensions do not match");
    }

    // Calculate the total count in each bin across all blocks.
    std::vector<uint64_t> bin_totals(num_bins, 0);

    for (size_t block = 0; block < num_blocks; ++block) {
        for (size_t bin = 0; bin < num_bins; ++bin) {
            bin_totals[bin] += hist[block * num_bins + bin];
        }
    }

    // Check that the total count fits in uint32_t.
    uint64_t total = 0;
    const uint64_t max_count = std::numeric_limits<uint32_t>::max();

    for (size_t bin = 0; bin < num_bins; ++bin) {
        if (bin_totals[bin] > max_count - total) {
            throw std::overflow_error("Total count exceeds uint32_t capacity");
        }
        total += bin_totals[bin];
    }

    std::vector<uint32_t> offsets(hist.size());

    // Count all elements belonging to lower-numbered bins.
    uint64_t lower_bin_count = 0;

    for (size_t bin = 0; bin < num_bins; ++bin) {
        uint64_t earlier_block_count = 0;

        for (size_t block = 0; block < num_blocks; ++block) {
            offsets[bin * num_blocks + block] =
                static_cast<uint32_t>(
                    lower_bin_count + earlier_block_count
                );

            earlier_block_count += hist[block * num_bins + bin];
        }

        lower_bin_count += bin_totals[bin];
    }

    return offsets;
}