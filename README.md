# Fast GPU Sorting

This project implements radix sort in CUDA for sorting unsigned 32-bit integers in ascending order. It compares the performance of our implementation with the CUB sorting library and a Futhark library implementation.

The default configuration uses 256 threads per block (`B=256`), 22 elements per thread (`Q=22`), and 8 bits per sorting pass (`r=8`). Each block processes up to 5,632 elements. There are 256 bins per pass, and four passes are needed to sort all 32 bits.

## CUDA Implementation

Each sorting pass consists of three main stages:

- **Histogram:** each block counts how many of its elements belong to each digit bin.
- **Global offsets:** the block histograms are transposed and scanned to calculate where each block's elements should start in the output array.
- **Stable scatter:** each element is written to its output position using the global offsets and its position among earlier elements with the same digit.

The scatter uses warp-level operations to calculate these positions while preserving the order of elements with the same digit. The output of one pass becomes the input of the next pass.

These are three logical stages, not three kernel functions. The global-offset stage uses separate transpose and scan kernels, and larger scans require multiple kernel launches.

## Code Structure

`main.cu` runs the complete CUDA radix sort on a small example or on numbers provided through the command line. It also checks the result against CPU `std::sort`.

`src/` contains the CUDA implementations:

- `block_histogram.cu` builds the block histograms.
- `scatter.cu` computes stable positions and writes the elements to the output.
- `radix_sort.cu` connects all three stages and runs the four sorting passes.
- `radix_config.cu` is retained for compatibility; the configuration is defined in its header.

`include/` contains the function declarations and shared configuration. `radix_config.h` defines B, Q and r. The headers in `include/global_offsets/` implement histogram transpose and exclusive scan. The integrated sorter currently requires `r=8`.

`tests/` contains the histogram, transpose, scan, global-offset, scatter and complete-sort tests. `tests/reference/` contains their CPU reference implementations. The tests include empty inputs, repeated values, boundary sizes and larger arrays.

`benchmarks/` contains programs for measuring the complete sort and individual components. It also contains the CUDA/CUB comparison script.

`experiments/` contains an alternative histogram implementation with separate counters for each warp. It did not improve performance in our measurements, so the original histogram remains in the complete sort.

`futhark-baseline/` contains a short program that calls the Futhark sorting library, together with a dataset generator and validation script.

`results/` contains saved benchmark measurements, validation logs and hardware information.

`tools/` contains optional programs for generating random and ordered input values.

`Literature/` contains the project specification, sorting papers and NVIDIA slides. `docs/` contains development and experiment notes; some historical commands there use the previous folder layout.

`Makefile` provides the build and test commands. Generated executables are placed in `build/`, which is excluded from Git.

## How to Compile and Run

The project requires Linux, GNU Make, a C++17 compiler and CUDA. The default build targets NVIDIA A100 GPUs using `-arch=sm_80`.

Extract the submitted ZIP and open a terminal inside the extracted project folder containing Makefile. Run all commands below from that folder.

```bash
module load cuda
make
./build/radix_sort
```

To sort your own unsigned 32-bit integers:

```bash
./build/radix_sort 7 0 4294967295 3 7 2147483648
```

The program prints the sorted values and checks them against the CPU result. This example does not measure sorting performance.

- `make` compiles the main program.
- `make run` compiles and runs the built-in example.
- `make tests` compiles all test programs.
- `make test` compiles and runs the component and integration tests.
- `make test-large` runs the complete-sort tests with an additional large dataset.
- `make test-cpu` runs the CPU offset test without requiring CUDA.
- `make benchmarks` compiles all CUDA benchmark programs without running them.
- `make tools` compiles the optional input generators.
- `make clean` removes the generated `build/` folder. It does not delete saved results.

After changing B or Q in `include/radix_config.h`, rebuild and validate the code:

```bash
make clean
make
make test
make test-large
```

## Validation

The component tests compare GPU outputs with CPU references. Complete-sort tests compare with `std::sort`, and digit-pass tests compare with `std::stable_sort`.

The reorganized implementation passed the main example, `make test` and `make test-large` on Hendrix with an NVIDIA A100 and the default configuration. All CUDA benchmark targets also compiled successfully. The saved logs are in `results/reorganized*.txt`.

To save a new validation run:

```bash
mkdir -p results
set -o pipefail
make test 2>&1 | tee results/validation_new.txt
make test-large 2>&1 | tee results/validation_large_new.txt
```

Use different filenames to preserve previous runs.

## CUDA and CUB Benchmarks

The full-sort benchmarks use 10K, 100K and 1M elements with random, all-equal, few-distinct, sorted and reversed input patterns. Both CUDA and CUB use five warm-up runs and 20 measurements, reporting the median time in milliseconds.

Timing uses CUDA events. Allocation, input restoration, CPU/GPU transfers and correctness checks are outside the timed section. Component benchmarks measure only their own stage; some use batches of calls to measure short operations.

Compile first, then run the executables directly to keep build messages out of the CSV files:

```bash
make benchmarks
mkdir -p results

./build/benchmark_cuda_sort > results/cuda_full_sort_new.csv 2> results/cuda_full_sort_new.log
./build/benchmark_cub_sort > results/cub_full_sort_new.csv 2> results/cub_full_sort_new.log

python3 benchmarks/compare_full_sort.py results/cub_full_sort_new.csv results/cuda_full_sort_new.csv > results/full_sort_comparison_new.csv
```

In the comparison file, `cuda_over_cub` is our CUDA time divided by CUB time. A value greater than one means CUB is faster.

The individual components can be benchmarked with:

```bash
./build/benchmark_histogram > results/histogram_new.csv 2> results/histogram_new.log
./build/benchmark_histogram_compare > results/histogram_warp_new.csv 2> results/histogram_warp_new.log
./build/scatter_benchmark > results/scatter_new.csv 2> results/scatter_new.log

./build/benchmark_global_offsets
./build/benchmark_workspace
./build/benchmark_offsets_cub
./build/benchmark_offset_tuning
```

Use the same GPU and input configuration when comparing results. Preserve repeated measurements when timings vary. Earlier scatter runs showed variation in the first few cases; these results should not be treated as proof of a speedup. Changing the timing method is also not an algorithm optimization.

## Futhark Baseline

`futhark-baseline/sort_u32.fut` calls radix sort from `github.com/diku-dk/sorts`. The package version is fixed in `futhark.pkg` and downloaded by `futhark pkg sync`.

The recorded baseline uses Futhark 0.25.24 and sorting package version 0.7.3. It uses the same input sizes and patterns as the CUDA and CUB benchmarks. The generated output is checked against CPU-sorted reference data.

From the project root on Hendrix, run these commands sequentially:

```bash
module load cuda
module load futhark
mkdir -p results

make futhark-build
make futhark-data

set -o pipefail
make futhark-validate 2>&1 | tee results/futhark_validation_new.txt
make futhark-bench 2>&1 | tee results/futhark_benchmark_new.txt
```

- `make futhark-build` downloads the pinned package and compiles the standalone sorting program.
- `make futhark-data` generates 15 input datasets and their CPU-sorted references.
- `make futhark-validate` rebuilds the standalone program, generates the data and checks all 15 datasets.
- `make futhark-bench` benchmarks the CUDA backend without tuning and saves raw timings in `results/futhark_sort_raw_new.json`.

The benchmark requests at least 20 measurements and also uses a minimum measurement duration. Futhark prints mean runtimes; use medians from the raw samples when comparing with CUDA/CUB medians. Raw times are in microseconds, so divide by 1,000 to obtain milliseconds. Document differences in timing methods when presenting comparisons.

`futhark bench` replaces the standalone executable with a server executable. Running `make futhark-validate` rebuilds the standalone version when needed.

The dataset generator links with `-lstdc++fs` for the Hendrix compiler. On a newer compiler where this library is unavailable, use:

```bash
make futhark-data FS_LIBS=
```

## Optional Input Generators

```bash
make tools
./build/generate_random_ints 100
./build/generate_sequence 0 100
```

For the separate scatter demonstration:

```bash
make build/scatter_demo
./build/scatter_demo --help
```

## References

The project specification and supplied reading are available in `Literature/`:

- [Project instructions](Literature/Project-Sorting.pdf).
- Nadathur Satish, Mark Harris and Michael Garland, [Designing Efficient Sorting Algorithms for Manycore GPUs](Literature/recommanded-paper-sorting.pdf).
- Linh Ha, Jens Krüger and Cláudio T. Silva, [Implicit radix sorting on GPUs](Literature/another-paper-sorting.pdf).
- [NVIDIA sorting slides](Literature/nvidia-sort-slides.pdf).

The report explains the algorithms, work–depth analysis, GPU memory use, validation and performance evaluation. This README describes how to build and reproduce the implementation.

## LLM Declaration

We used ChatGPT/Codex to help draft and revise parts of the code,and debug errors.
The group reviewed and adapted the generated suggestions and tested the implementation against CPU references on an NVIDIA A100 GPU. We take responsibility for the submitted code and reported results.
