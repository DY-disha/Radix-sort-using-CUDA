NVCC ?= nvcc
CXX ?= g++
ARCH ?= sm_80
NVCCFLAGS ?= -O2 -std=c++17 -lineinfo
CXXFLAGS ?= -O2 -std=c++17
INCLUDES := -Iinclude -Itests/reference -Iexperiments
HEADERS := $(wildcard include/*.h include/*.cuh include/global_offsets/*.cuh tests/reference/*.h tests/reference/*.hpp experiments/*.cuh)
CORE := src/block_histogram.cu src/scatter.cu src/radix_sort.cu
TEST_NAMES := histogram_test integration_test scatter_test radix_sort_test test_transpose test_scan test_global_offsets scatter_demo
TEST_PROGRAMS := $(addprefix build/,$(TEST_NAMES))
BENCH_NAMES := benchmark_cuda_sort benchmark_cub_sort benchmark_histogram benchmark_histogram_compare scatter_benchmark benchmark_global_offsets benchmark_workspace benchmark_offsets_cub benchmark_offset_tuning
BENCH_PROGRAMS := $(addprefix build/,$(BENCH_NAMES))
.PHONY: all run tests test test-large test-cpu benchmarks benchmark-scatter tools clean futhark-build futhark-data futhark-validate futhark-bench
all: build/radix_sort
run: build/radix_sort
	./build/radix_sort
build:
	mkdir -p build
build/radix_sort: main.cu $(CORE) $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) main.cu $(CORE) -o $@
build/histogram_test: src/block_histogram.cu tests/test_histogram.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) src/block_histogram.cu tests/test_histogram.cu -o $@
build/integration_test: src/block_histogram.cu tests/test_histogram_offsets.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) src/block_histogram.cu tests/test_histogram_offsets.cu -o $@
build/scatter_test: src/scatter.cu tests/test_scatter_unit.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) src/scatter.cu tests/test_scatter_unit.cu -o $@
build/radix_sort_test: $(CORE) tests/test_radix_sort.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) $(CORE) tests/test_radix_sort.cu -o $@
build/scatter_demo: src/scatter.cu tests/test_scatter.cpp tests/reference/radix_sort_cpu.cpp $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) src/scatter.cu tests/test_scatter.cpp tests/reference/radix_sort_cpu.cpp -o $@
build/benchmark_cuda_sort: $(CORE) benchmarks/benchmark_cuda_sort.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) $(CORE) benchmarks/benchmark_cuda_sort.cu -o $@
build/benchmark_cub_sort: benchmarks/benchmark_cub_sort.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) benchmarks/benchmark_cub_sort.cu -o $@
build/benchmark_histogram: src/block_histogram.cu benchmarks/benchmark_histogram.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) src/block_histogram.cu benchmarks/benchmark_histogram.cu -o $@
build/benchmark_histogram_compare: src/block_histogram.cu experiments/block_histogram_warp.cu benchmarks/benchmark_histogram_compare.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) src/block_histogram.cu experiments/block_histogram_warp.cu benchmarks/benchmark_histogram_compare.cu -o $@
build/scatter_benchmark: src/scatter.cu benchmarks/benchmark_scatter.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) src/scatter.cu benchmarks/benchmark_scatter.cu -o $@
build/test_transpose: tests/test_transpose.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) tests/test_transpose.cu -o $@
build/test_scan: tests/test_scan.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) tests/test_scan.cu -o $@
build/test_global_offsets: tests/test_global_offsets.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) tests/test_global_offsets.cu -o $@
build/benchmark_global_offsets: benchmarks/benchmark_global_offsets.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) benchmarks/benchmark_global_offsets.cu -o $@
build/benchmark_workspace: benchmarks/benchmark_workspace.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) benchmarks/benchmark_workspace.cu -o $@
build/benchmark_offsets_cub: benchmarks/benchmark_offsets_cub.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) benchmarks/benchmark_offsets_cub.cu -o $@
build/benchmark_offset_tuning: benchmarks/benchmark_offset_tuning.cu $(HEADERS) Makefile | build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(INCLUDES) benchmarks/benchmark_offset_tuning.cu -o $@
build/test_cpu: tests/test_cpu.cpp $(HEADERS) Makefile | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) tests/test_cpu.cpp -o $@
build/generate_random_ints: tools/generate_random_ints.cpp $(HEADERS) Makefile | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) tools/generate_random_ints.cpp -o $@
build/generate_sequence: tools/generate_sequence.cpp $(HEADERS) Makefile | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) tools/generate_sequence.cpp -o $@
tests: $(TEST_PROGRAMS) build/test_cpu
test-cpu: build/test_cpu
	./build/test_cpu
test: tests
	./build/test_cpu
	./build/histogram_test
	./build/integration_test
	./build/test_transpose
	./build/test_scan
	./build/test_global_offsets
	./build/scatter_test
	./build/scatter_demo
	./build/radix_sort_test
test-large: build/radix_sort_test
	./build/radix_sort_test --large
benchmarks: $(BENCH_PROGRAMS)
benchmark-scatter: build/scatter_benchmark
	./build/scatter_benchmark
tools: build/generate_random_ints build/generate_sequence
# Run each Futhark target separately; bench replaces the standalone executable.
futhark-build:
	cd futhark-baseline && futhark pkg sync && futhark check sort_u32.fut && futhark cuda sort_u32.fut
FS_LIBS ?= -lstdc++fs
futhark-baseline/generate_sort_data: futhark-baseline/generate_sort_data.cpp Makefile
	$(CXX) $(CXXFLAGS) $< -o $@ $(FS_LIBS)
futhark-data: futhark-baseline/generate_sort_data
	cd futhark-baseline && ./generate_sort_data
futhark-validate: futhark-build futhark-data
	cd futhark-baseline && python3 validate_sort.py
futhark-bench: futhark-data
	mkdir -p results
	cd futhark-baseline && futhark pkg sync && futhark bench --backend=cuda --no-tuning --runs=20 --no-convergence-phase --spec-file=datasets.spec --json=../results/futhark_sort_raw_new.json sort_u32.fut
clean:
	rm -rf build
