#!/usr/bin/env python3
"""Join verified CUDA/CUB CSV timing outputs; refuse missing or failed cases."""
import csv
import sys
from pathlib import Path

FIELDS = ("implementation", "n", "pattern", "median_ms", "status")

def load(path, expected_impl):
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing required headers")
        for row in reader:
            if row["implementation"] != expected_impl or row["status"] != "PASS":
                raise ValueError(f"{path}: non-{expected_impl} or failed row: {row}")
            key = (int(row["n"]), row["pattern"])
            if key in result:
                raise ValueError(f"{path}: duplicate case {key}")
            ms = float(row["median_ms"])
            if ms <= 0:
                raise ValueError(f"{path}: invalid median {ms}")
            result[key] = ms
    return result

def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python compare_full_sort.py cub_full_sort_results.csv cuda_full_sort_results.csv")
    cub = load(sys.argv[1], "CUB")
    cuda = load(sys.argv[2], "CUDA")
    if cub.keys() != cuda.keys():
        missing_cub = sorted(cuda.keys() - cub.keys())
        missing_cuda = sorted(cub.keys() - cuda.keys())
        raise ValueError(f"Mismatched cases: missing CUB={missing_cub}; missing CUDA={missing_cuda}")
    writer = csv.writer(sys.stdout)
    writer.writerow(["n", "pattern", "cuda_ms", "cub_ms", "cuda_over_cub"])
    for n, pattern in sorted(cub):
        writer.writerow([n, pattern, f"{cuda[n, pattern]:.6f}",
                         f"{cub[n, pattern]:.6f}", f"{cuda[n, pattern]/cub[n, pattern]:.3f}"])

if __name__ == "__main__":
    main()
