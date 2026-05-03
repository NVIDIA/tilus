# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark all hopper_matmul versions against cuBLAS (torch.matmul).

Run directly:
    python benchmark.py

Or via the slurm job:
    sbatch sample_slurm_hopper_benchmark.sh
"""

import importlib
import math
import sys

import pandas
import torch

import tilus
from tilus.utils import benchmark_func, cdiv

tilus.option.cache_dir("./cache")

WORKLOADS = [
    # (m, n, k, label)
    (1024,  1024,  1024,  "1k-sq"),
    (2048,  2048,  2048,  "2k-sq"),
    (4096,  4096,  4096,  "4k-sq"),
    (4096,  4096,  14336, "llm-ffn"),
    (8192,  8192,  8192,  "8k-sq"),
    (10240, 10240, 10240, "10k-sq"),
]

VERSIONS = ["v0", "v1", "v2", "v3", "v4", "v5"]

VERSION_CLASS = {
    "v0": "MatmulTMA",
    "v1": "MatmulWGMMA",
    "v2": "MatmulWGMMAV2",
    "v3": "MatmulWGMMAV3",
    "v4": "MatmulWGMMAV4",
    "v5": "MatmulWGMMAV5",
}

WARMUP = 5
REPEAT = 30


def load_version(name: str):
    mod = importlib.import_module(f"matmul_{name}")
    return getattr(mod, VERSION_CLASS[name])


def tflops(m, n, k, latency_ms):
    return 2 * m * n * k / latency_ms * 1e-9


def run_benchmark(versions=None):
    if versions is None:
        versions = VERSIONS

    device = torch.cuda.get_device_name(0)
    print(f"Device: {device}")
    print(f"Versions: {versions}")
    print()

    headers = ["workload", "m", "n", "k", "kernel", "latency (ms)", "tflops", "% of cublas"]
    rows = []

    for m, n, k, label in WORKLOADS:
        print(f"--- {label}  m={m} n={n} k={k} ---")
        a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        b = (torch.rand(n, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
        c_ref = torch.empty(m, n, dtype=torch.float16).cuda()
        c_tilus = torch.empty(m, n, dtype=torch.float16).cuda()

        # cuBLAS baseline
        cublas_lat = benchmark_func(lambda: torch.matmul(a, b.T, out=c_ref), warmup=WARMUP, repeat=REPEAT)
        cublas_tf = tflops(m, n, k, cublas_lat)
        rows.append([label, m, n, k, "cublas", cublas_lat, cublas_tf, 100.0])
        print(f"  cublas  {cublas_lat:.4f} ms  {cublas_tf:.1f} TFLOPS")

        for ver in versions:
            try:
                cls = load_version(ver)
                kernel = cls()
                # correctness check
                kernel(m, n, k, a, b, c_tilus)
                torch.cuda.synchronize()
                torch.testing.assert_close(c_ref, c_tilus, atol=1e-2, rtol=1e-2)

                lat = benchmark_func(lambda: kernel(m, n, k, a, b, c_tilus), warmup=WARMUP, repeat=REPEAT)
                tf = tflops(m, n, k, lat)
                pct = tf / cublas_tf * 100.0
                rows.append([label, m, n, k, f"tilus-{ver}", lat, tf, pct])
                print(f"  tilus-{ver}  {lat:.4f} ms  {tf:.1f} TFLOPS  ({pct:.1f}% of cuBLAS)")
            except Exception as e:
                print(f"  tilus-{ver}  ERROR: {e}", file=sys.stderr)
                rows.append([label, m, n, k, f"tilus-{ver}", float("nan"), float("nan"), float("nan")])

        print()

    df = pandas.DataFrame(rows, columns=headers)
    print("\n=== Summary ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", default=None,
                        help="Subset of versions to benchmark, e.g. --versions v3 v4 v5")
    args = parser.parse_args()
    run_benchmark(versions=args.versions)
