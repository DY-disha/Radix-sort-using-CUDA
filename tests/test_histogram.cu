#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using Key = unsigned int;

// Function declarations.
// Their implementations are in block_histogram.cu.
unsigned threads_per_block();
unsigned elements_per_thread();
size_t tile_size();

std::vector<Key> histogram_cpu(
    const std::vector<Key>& input,
    unsigned shift,
    unsigned r);

std::vector<Key> histogram_gpu(
    const std::vector<Key>& input,
    unsigned shift,
    unsigned r);

void check_sums(
    const std::vector<Key>& hist,
    size_t n,
    unsigned H);

int main()
{
    try {
        const size_t TILE = tile_size();

        // Test 1: manually calculated examples.
        auto low = histogram_cpu({7, 6, 5, 4}, 0, 2);
        auto high = histogram_cpu({7, 6, 5, 4}, 2, 2);

        if (low != std::vector<Key>{1, 1, 1, 1} ||
            high != std::vector<Key>{0, 4, 0, 0}) {
            throw std::runtime_error(
                "CPU reference failed manual example");
        }

        // Test 2: one full tile plus one extra element.
        std::vector<Key> boundary(TILE + 1, 0);
        boundary.back() = 3;

        auto boundary_hist = histogram_cpu(boundary, 0, 2);

        if (boundary_hist !=
            std::vector<Key>{Key(TILE), 0, 0, 0, 0, 0, 0, 1}) {
            throw std::runtime_error(
                "CPU reference failed tile-boundary example");
        }

        // Test 3: compare GPU and CPU across many cases.
        std::mt19937 rng(42);
        size_t tests = 0;

        std::vector<size_t> sizes = {
            0, 1, 7,
            TILE - 1,
            TILE,
            TILE + 1,
            3 * TILE + 17
        };

        for (size_t n : sizes) {
            for (unsigned pattern = 0; pattern < 4; ++pattern) {
                std::vector<Key> input(n);

                for (size_t i = 0; i < n; ++i) {
                    if (pattern == 0) {
                        input[i] = Key(rng());  // Random
                    }
                    else if (pattern == 1) {
                        input[i] = ~0u;        // All identical
                    }
                    else if (pattern == 2) {
                        input[i] = Key(i);     // Increasing
                    }
                    else {
                        input[i] = Key(n - i); // Decreasing
                    }
                }

                for (unsigned r : {1u, 2u, 4u, 8u}) {
                    for (unsigned shift = 0;
                         shift < 32;
                         shift += r) {
                        auto expected =
                            histogram_cpu(input, shift, r);

                        auto actual =
                            histogram_gpu(input, shift, r);

                        if (actual != expected) {
                            std::cerr
                                << "Mismatch: N=" << n
                                << " pattern=" << pattern
                                << " r=" << r
                                << " shift=" << shift << '\n';

                            return 1;
                        }

                        check_sums(expected, n, 1u << r);
                        check_sums(actual, n, 1u << r);

                        ++tests;
                    }
                }
            }
        }

        std::cout
            << "GPU HISTOGRAM VALID: "
            << tests
            << " cases match CPU reference.\n";

        std::cout
            << "B=" << threads_per_block()
            << " Q=" << elements_per_thread()
            << " TILE=" << TILE
            << "; layout hist[block * H + bin], "
            << "unsigned 32-bit counts\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}