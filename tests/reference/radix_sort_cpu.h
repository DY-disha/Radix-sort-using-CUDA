#pragma once

#include <cstdint>
#include <vector>

std::vector<uint32_t> radixSortPassCPU(
    const std::vector<uint32_t>& input,
    unsigned radixBits,
    unsigned passNumber
);