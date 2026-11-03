// Build from project root with block_histogram.cu (NOT main.cu).
#include "global_offsets/global_offsets.cuh"
#include "offset_reference.hpp"
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>
using Key = unsigned int;
static_assert(std::is_same<Key, uint32_t>::value, "Histogram and offset types must match");
size_t tile_size();
std::vector<Key> histogram_cpu(const std::vector<Key>&, unsigned, unsigned);
cudaError_t histogram_device(const Key*, size_t, unsigned, unsigned, Key*, cudaStream_t);

void require_cuda(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
// Owns GPU memory, including cleanup if a test fails.
struct Buffer {
    uint32_t* p = nullptr;
    explicit Buffer(size_t n) {
        if (n) require_cuda(cudaMalloc(reinterpret_cast<void**>(&p), n*sizeof(uint32_t)));
    }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
void compare_gpu(const Buffer& data, const std::vector<uint32_t>& expected,
                 const std::string& label) {
    std::vector<uint32_t> actual(expected.size());
    if (!actual.empty()) require_cuda(cudaMemcpy(actual.data(), data.p,
        actual.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (size_t i=0; i<actual.size(); ++i)
        if (actual[i] != expected[i])
            throw std::runtime_error(label + " mismatch at " + std::to_string(i));
}
void run_case(const std::vector<Key>& input, unsigned r, unsigned shift) {
    const size_t T = tile_size();
    const size_t G = input.size()/T + (input.size()%T != 0);
    const size_t H = size_t(1) << r;
    const size_t entries = G*H;
    auto expected_hist = histogram_cpu(input, shift, r);
    auto expected_offsets = compute_offsets_cpu(expected_hist, G, H);
    Buffer keys(input.size()), hist(entries), transposed(entries), offsets(entries);
    size_t workspace_size = offset_scan::workspace_elements_required(entries);
    Buffer workspace(workspace_size);
    if (!input.empty()) require_cuda(cudaMemcpy(keys.p, input.data(),
        input.size()*sizeof(Key), cudaMemcpyHostToDevice));

    // Actual integration: histogram -> offsets entirely on the GPU.
    require_cuda(histogram_device(keys.p, input.size(), shift, r, hist.p, 0));
    require_cuda(compute_global_offsets(hist.p, transposed.p, offsets.p, G, H));
    require_cuda(cudaDeviceSynchronize());
    compare_gpu(hist, expected_hist, "histogram");
    compare_gpu(offsets, expected_offsets, "original offsets");

    // Reuse the GPU histogram with Anika's workspace implementation.
    require_cuda(compute_global_offsets_with_workspace(hist.p, transposed.p,
        offsets.p, G, H, workspace.p, workspace_size));
    require_cuda(cudaDeviceSynchronize());
    compare_gpu(offsets, expected_offsets, "workspace offsets");
}
int main() {
    try {
        // Independently hand-checked offset layout example.
        if (compute_offsets_cpu({2,1,1,2},2,2) !=
            std::vector<uint32_t>{0,2,3,4})
            throw std::runtime_error("Manual offset example failed");
        const size_t T=tile_size();
        const std::vector<size_t> sizes={0,1,T-1,T,T+1,2*T,2*T+1,33*T+17};
        std::mt19937 rng(42);
        size_t passed=0;
        for (size_t n:sizes) {
            for (unsigned pattern=0; pattern<3; ++pattern) {
                std::vector<Key> input(n);
                for (size_t i=0;i<n;++i)
                    input[i]=pattern==0 ? Key(rng()) : pattern==1 ? ~0u : Key(i%7);
                for (unsigned r:{1u,2u,4u,8u}) {
                    for (unsigned shift:{0u,32-r}) {
                        try { run_case(input,r,shift); }
                        catch (const std::exception& e) {
                            throw std::runtime_error("N="+std::to_string(n)+
                                " pattern="+std::to_string(pattern)+" r="+std::to_string(r)+
                                " shift="+std::to_string(shift)+": "+e.what());
                        }
                        ++passed;
                    }
                }
            }
        }
        std::cout << "INTEGRATION VALID: " << passed
                  << " cases; histogram and both offset implementations match CPU references.\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n'; return 1;
    }
}
