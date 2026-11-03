
#include "radix_sort.cuh"
#include "radix_config.h"
#include "block_histogram.cuh"
#include "global_offsets/global_offsets.cuh"
#include "scatter.cuh"

#include <stdexcept>
#include <type_traits>
#include <utility>

static_assert(std::is_same<uint32_t, unsigned int>::value);

namespace {

void checked(cudaError_t e) {
    if (e != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(e));
    }
}

void allocate(uint32_t*& p, size_t count) {
    if (count) {
        checked(cudaMalloc(
            reinterpret_cast<void**>(&p),
            count * sizeof(uint32_t)
        ));
    }
}

} // namespace

RadixWorkspace::RadixWorkspace(size_t n)
    : n_(n), blocks_(0), scan_elements_(0) {

    if (n > UINT32_MAX) {
        throw std::invalid_argument(
            "32-bit offsets require N <= UINT32_MAX"
        );
    }

    blocks_ = n / radix_config::TILE
        + (n % radix_config::TILE != 0);

    const size_t entries =
        blocks_ * radix_config::NUM_BINS;

    scan_elements_ =
        offset_scan::workspace_elements_required(entries);

    try {
        allocate(hist_, entries);
        allocate(transposed_, entries);
        allocate(offsets_, entries);
        allocate(scan_, scan_elements_);
    } catch (...) {
        release();
        throw;
    }
}

void RadixWorkspace::release() noexcept {
    if (hist_) cudaFree(hist_);
    if (transposed_) cudaFree(transposed_);
    if (offsets_) cudaFree(offsets_);
    if (scan_) cudaFree(scan_);
}

RadixWorkspace::~RadixWorkspace() {
    release();
}

void radix_pass_device(
    const uint32_t* input,
    uint32_t* output,
    size_t n,
    unsigned shift,
    RadixWorkspace& w,
    cudaStream_t stream
) {
    if (n != w.n_ ||
        shift >= 32 ||
        shift % radix_config::RADIX_BITS != 0) {

        throw std::invalid_argument(
            "Workspace size or radix shift mismatch"
        );
    }

    if (!n) return;

    if (!input || !output || input == output) {
        throw std::invalid_argument(
            "Pass requires separate non-null key buffers"
        );
    }

    // Stage 1: Compute block histograms.
    checked(histogram_device(
        input,
        n,
        shift,
        radix_config::RADIX_BITS,
        w.hist_,
        stream
    ));

    // Stage 2: Multi-block global offsets.
    // Experimental optimization: use multiple CUDA blocks
    // for the histogram transpose while retaining the
    // workspace-based hierarchical scan.
    checked(compute_global_offsets_multiblock(
        w.hist_,
        w.transposed_,
        w.offsets_,
        w.blocks_,
        radix_config::NUM_BINS,
        w.scan_,
        w.scan_elements_,
        stream
    ));

    // Stage 3: Stable scatter.
    checked(launch_stable_scatter(
        input,
        output,
        w.offsets_,
        n,
        int(shift),
        stream
    ));
}

uint32_t* radix_sort_device(
    uint32_t* input,
    uint32_t* scratch,
    size_t n,
    RadixWorkspace& w,
    cudaStream_t stream
) {
    for (unsigned shift = 0;
         shift < 32;
         shift += radix_config::RADIX_BITS) {

        radix_pass_device(
            input,
            scratch,
            n,
            shift,
            w,
            stream
        );

        std::swap(input, scratch);
    }

    return input;
}
