# Per-warp histogram experiment

Configuration: B=256, Q=22, r=8, 256 bins.
GPU: NVIDIA A100-PCIE-40GB.

Baseline: one shared histogram per block.
Experiment: one histogram per warp, followed by combining
the eight histograms into the existing block-major output.

Validation:
- 4,320 boundary/reference checks passed.
- Both implementations passed all 60 benchmark cases.

Timing:
- At least one second of startup warm-up.
- 500 warm-up calls per implementation for each case.
- 20 measurements per implementation, each batching 100 calls.
- Measurement order alternates between implementations.
- Allocation, copies, and validation are excluded.
- Each shift uses the original input order.

Results:
The experimental implementation took approximately 0.8–3.9%
longer in this comparison run. No improvement was observed.
These are standalone histogram measurements, not full-sort timings.

Decision:
Keep block_histogram.cu as the production implementation.
Retain block_histogram_warp.cu for reference and further experiments.

Results:
results/histogram_warp_comparison.csv
results/histogram_warp_comparison.log