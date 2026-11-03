
#pragma once

#include <cstdint>
#include <cstddef>
#include <climits>
#include <stdexcept>
#include <vector>

#include "radix_config.h"

// Generates the offsets required by stable_scatter_kernel.
//
// input:       Input array for the current radix pass.
// shift:       0, 8, 16, or 24.
// num_blocks:  Number of tiles in the input.
// offsets:     Output array indexed as [digit * num_blocks + block].
//
// The function is CPU-only and is intended as a reference
// implementation for testing Person 3's CUDA scatter kernel.
inline void generate_reference_offsets(
    const std::vector<uint32_t>& input,
    int shift,
    int& num_blocks,
    std::vector<uint32_t>& offsets)
{
    if (shift != 0 && shift != 8 &&
        shift != 16 && shift != 24)
    {
        throw std::invalid_argument("shift must be 0, 8, 16, or 24");
    }

    const size_t n = input.size();

    // Calculate the number of blocks, including a partial final block.
    const size_t blocks =
        (n + OFFSET_ITEMS_PER_BLOCK - 1) / OFFSET_ITEMS_PER_BLOCK;

    if (blocks > static_cast<size_t>(INT_MAX))
        throw std::overflow_error("Too many blocks");

    if (n > UINT32_MAX)
        throw std::overflow_error(
            "Input too large for uint32_t offsets");

    num_blocks = static_cast<int>(blocks);

    // No elements means no offsets are needed.
    if (n == 0)
    {
        offsets.clear();
        return;
    }

    // Histogram layout: hist[block][digit].
    std::vector<uint32_t> hist(
        blocks * OFFSET_NUM_BINS, 0);

    // Step 1: Build one histogram per block.
    for (size_t i = 0; i < n; ++i)
    {
        const size_t block = i / OFFSET_ITEMS_PER_BLOCK;

        const uint32_t digit =
            (input[i] >> shift) & (OFFSET_NUM_BINS - 1);

        ++hist[block * OFFSET_NUM_BINS + digit];
    }

    // Offset layout must match the CUDA scatter kernel:
    // offsets[digit * num_blocks + block].
    offsets.assign(blocks * OFFSET_NUM_BINS, 0);

    // Step 2: Calculate the starting offset of each digit.
    uint32_t digit_base = 0;

    for (int digit = 0; digit < OFFSET_NUM_BINS; ++digit)
    {
        // Within this digit, blocks are ordered by input position.
        uint32_t block_offset = digit_base;

        for (size_t block = 0; block < blocks; ++block)
        {
            offsets[
                static_cast<size_t>(digit) * blocks + block
            ] = block_offset;

            block_offset +=
                hist[block * OFFSET_NUM_BINS + digit];
        }

        // The next digit begins after all elements of this digit.
        digit_base = block_offset;
    }

    // For a valid histogram, digit_base must equal the input size.
    if (digit_base != n)
        throw std::runtime_error("Offset generation failed");
}
