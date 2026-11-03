#pragma once
#include <cstddef>

// Shared sorting configuration. Rebuild all targets after changing it.
namespace radix_config {
inline constexpr unsigned B = 256;
inline constexpr unsigned Q = 22;
inline constexpr unsigned RADIX_BITS = 8;
inline constexpr unsigned NUM_BINS = 1u << RADIX_BITS;
inline constexpr size_t TILE = size_t(B) * Q;
static_assert(B > 0 && B <= 1024 && B % 32 == 0);
static_assert(Q > 0);
static_assert(RADIX_BITS == 8, "Integrated scatter currently supports 8-bit digits");
}
// Compatibility names used by Maks's existing tests.
inline constexpr int OFFSET_RADIX_BITS = radix_config::RADIX_BITS;
inline constexpr int OFFSET_NUM_BINS = radix_config::NUM_BINS;
inline constexpr int OFFSET_ITEMS_PER_BLOCK = int(radix_config::TILE);
