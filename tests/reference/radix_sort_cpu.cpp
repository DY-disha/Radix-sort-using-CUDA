#include "radix_sort_cpu.h"

#include <stdexcept>
#include <limits>

struct RadixConfig {
    unsigned radixBits;
    unsigned numBins;
    unsigned itemsPerBlock;
};

std::vector<uint32_t> radixSortPassCPU(
    const std::vector<uint32_t>& input,
    unsigned radixBits,
    unsigned passNumber)
{
    constexpr unsigned KEY_BITS = 32;

    if (radixBits == 0 || radixBits > KEY_BITS) {
        throw std::invalid_argument(
            "radixBits must be between 1 and 32");
    }

    if (passNumber >= (KEY_BITS + radixBits - 1) / radixBits) {
        throw std::invalid_argument(
            "passNumber exceeds the number of passes");
    }

    const unsigned shift = passNumber * radixBits;
    const unsigned bitsThisPass =
        (radixBits < KEY_BITS - shift)
            ? radixBits
            : KEY_BITS - shift;

    // Use 64-bit arithmetic to avoid shifting 1u by 32.
    const uint64_t numBins = uint64_t{1} << bitsThisPass;
    const uint32_t mask =
        static_cast<uint32_t>(numBins - 1);

    std::vector<size_t> counts(
        static_cast<size_t>(numBins), 0);

    // 1. Histogram: count each digit.
    for (uint32_t value : input) {
        const uint32_t digit = (value >> shift) & mask;
        ++counts[digit];
    }

    // 2. Exclusive prefix sum: starting output position per digit.
    size_t offset = 0;

    for (size_t digit = 0; digit < counts.size(); ++digit) {
        const size_t count = counts[digit];
        counts[digit] = offset;
        offset += count;
    }

    // 3. Stable scatter.
    // Process input from left to right so equal digits retain
    // their relative order.
    std::vector<uint32_t> output(input.size());

    for (uint32_t value : input) {
        const uint32_t digit = (value >> shift) & mask;
        output[counts[digit]++] = value;
    }

    return output;
}