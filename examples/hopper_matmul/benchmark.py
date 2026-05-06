# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark all hopper_matmul versions against cuBLAS (torch.matmul).

Run directly:
    python benchmark.py
    python benchmark.py --ncu
    python benchmark.py --versions v3 v4 v5 --size 4096 4096 4096
"""

import argparse
import csv
import io
import subprocess
import time

VERSION_NAMES = ["v0", "v1", "v2", "v3", "v4", "v5"]

VERSION_CLASS = {
    "v0": "MatmulTMA",
    "v1": "MatmulWGMMA",
    "v2": "MatmulWGMMAV2",
    "v3": "MatmulWGMMAV3",
    "v4": "MatmulWGMMAV4",
    "v5": "MatmulWGMMAV5",
}


def _load_version(name: str):
    """Lazily import a matmul module by version name and return the class."""
    import importlib

    import tilus

    tilus.option.cache_dir("./cache")

    module = importlib.import_module(f"matmul_{name}")
    return getattr(module, VERSION_CLASS[name])


def run_kernels(version_names: list, m_size: int, n_size: int, k_size: int):
    """Run cuBLAS and tilus matmul versions sequentially (used as the target for ncu_run)."""
    import torch

    a = torch.randn(m_size, k_size, dtype=torch.float16, device="cuda")
    b = torch.randn(n_size, k_size, dtype=torch.float16, device="cuda")
    c = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")

    # tilus versions
    for name in version_names:
        matmul = _load_version(name)()
        matmul(m_size, n_size, k_size, a, b, c)
        torch.cuda.synchronize()

    # cuBLAS
    _ = a @ b.T
    torch.cuda.synchronize()


def _read_ncu_csv(
    report_path: str, page: str, metrics: str | None = None
) -> csv.DictReader:
    """Run ncu --import --csv and return a DictReader, skipping the units row."""
    cmd = ["/usr/local/cuda/bin/ncu", "--import", report_path, "--csv", "--page", page]
    if metrics:
        cmd += ["--metrics", metrics]
    result = subprocess.run(cmd, capture_output=True, text=True)
    reader = csv.DictReader(io.StringIO(result.stdout))
    next(reader, None)
    return reader


def _short_kernel_name(name: str) -> str:
    idx = name.find("(")
    return name[:idx] if idx != -1 else name


def parse_ncu_report(report_path: str) -> list[tuple[str, dict]]:
    """Extract per-kernel metrics from an NCU report. Returns [(kernel_name, metrics), ...] in order."""
    tensor_col = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
    reader = _read_ncu_csv(report_path, "raw", metrics=tensor_col)
    per_kernel: dict[str, dict] = {}
    kernel_order: list[str] = []
    for row in reader:
        kernel = _short_kernel_name(row["Kernel Name"])
        if kernel not in per_kernel:
            per_kernel[kernel] = {}
            kernel_order.append(kernel)
        metrics = per_kernel[kernel]
        if tensor_col in row and row[tensor_col]:
            metrics["tensor_core_util (%)"] = float(row[tensor_col])

    reader2 = _read_ncu_csv(report_path, "details")
    for row in reader2:
        kernel = _short_kernel_name(row["Kernel Name"])
        if kernel not in per_kernel:
            per_kernel[kernel] = {}
            kernel_order.append(kernel)
        metrics = per_kernel[kernel]
        if row.get("Metric Name") == "DRAM Throughput":
            metrics["dram_throughput (%)"] = float(row["Metric Value"])
        if row.get("Metric Name") == "Compute (SM) Throughput":
            metrics["sm_throughput (%)"] = float(row["Metric Value"])
        if row.get("Metric Name") == "SM Frequency":
            metrics["sm_freq (GHz)"] = float(row["Metric Value"])
        if row.get("Metric Name") == "Duration":
            value = float(row["Metric Value"])
            unit = row.get("Metric Unit", "ms")
            if unit == "us":
                value /= 1000.0
            elif unit == "s":
                value *= 1000.0
            metrics["duration (ms)"] = value

    return [(k, per_kernel[k]) for k in kernel_order]


def benchmark_all(versions: list[str], m_size: int, n_size: int, k_size: int):
    """Benchmark all versions using benchmark_func."""
    import math

    import pandas
    import torch
    from tilus.utils import benchmark_func

    headers = ["version", "latency (ms)", "tflops", "% of cublas"]
    rows = []

    a = (torch.rand(m_size, k_size, dtype=torch.float16, device="cuda") - 0.5) / math.sqrt(k_size)
    b = (torch.rand(n_size, k_size, dtype=torch.float16, device="cuda") - 0.5) / math.sqrt(k_size)
    c_ref = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")
    c_tilus = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")

    # cuBLAS baseline first, so we can compute % of cublas
    cublas_lat = benchmark_func(lambda: torch.matmul(a, b.T, out=c_ref), warmup=5, repeat=30)
    cublas_tf = 2 * m_size * n_size * k_size / cublas_lat * 1e-9

    for name in versions:
        try:
            matmul = _load_version(name)()
            # warmup + correctness check
            matmul(m_size, n_size, k_size, a, b, c_tilus)
            torch.cuda.synchronize()
            torch.testing.assert_close(c_ref, c_tilus, atol=1e-2, rtol=1e-2)

            latency = benchmark_func(
                lambda: matmul(m_size, n_size, k_size, a, b, c_tilus), warmup=5, repeat=30
            )
            tf = 2 * m_size * n_size * k_size / latency * 1e-9
            pct = tf / cublas_tf * 100.0
            rows.append([f"tilus_{name}", latency, tf, pct])
            time.sleep(1)  # cool down between runs
        except Exception as e:
            print(f"  tilus_{name}  ERROR: {e}")
            rows.append([f"tilus_{name}", float("nan"), float("nan"), float("nan")])

    rows.append(["cublas", cublas_lat, cublas_tf, 100.0])

    df = pandas.DataFrame(rows, columns=headers)
    print(f"\nBenchmark results (m={m_size}, n={n_size}, k={k_size}):")
    print(df.to_string(index=False))


def ncu_profile_all(versions: list[str], m_size: int, n_size: int, k_size: int):
    """Profile all versions in a single ncu_run and extract key metrics."""
    import pandas
    import tilus

    print("Warming up (JIT + autotuning)...")
    run_kernels(versions, m_size, n_size, k_size)

    labels = list(versions) + ["cublas"]

    print(f"Profiling cublas, {', '.join(versions)} ...")
    report = tilus.utils.ncu_run(
        run_kernels,
        versions,
        m_size,
        n_size,
        k_size,
        kernel_regex="tilus|cutlass|sm90|gemm|cublas",
    )
    print(f"Report saved to: {report.report_path}")

    kernel_metrics = parse_ncu_report(report.report_path)

    headers = [
        "version",
        "kernel",
        "duration (ms)",
        "tflops",
        "sm_freq (GHz)",
        "sm_throughput (%)",
        "dram_throughput (%)",
        "tensor_core_util (%)",
    ]
    rows = []
    for i, name in enumerate(labels):
        if i < len(kernel_metrics):
            kernel, metrics = kernel_metrics[i]
        else:
            kernel, metrics = "?", {}
        duration_ms = metrics.get("duration (ms)", "")
        tflops = 2 * m_size * n_size * k_size / duration_ms * 1e-9 if duration_ms else ""
        rows.append(
            [
                name,
                kernel,
                duration_ms,
                tflops,
                metrics.get("sm_freq (GHz)", ""),
                metrics.get("sm_throughput (%)", ""),
                metrics.get("dram_throughput (%)", ""),
                metrics.get("tensor_core_util (%)", ""),
            ]
        )

    df = pandas.DataFrame(rows, columns=headers)
    print(f"\nNCU profiling results (m={m_size}, n={n_size}, k={k_size}):")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Benchmark Hopper matmul V0-V5")
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Use NCU profiling instead of benchmark_func",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=VERSION_NAMES,
        choices=VERSION_NAMES,
        help="Which versions to benchmark (default: all)",
    )
    parser.add_argument(
        "--size",
        nargs=3,
        type=int,
        default=[8192, 8192, 8192],
        metavar=("M", "N", "K"),
        help="Workload size M N K (default: 8192 8192 8192)",
    )
    args = parser.parse_args()
    m_size, n_size, k_size = args.size

    if args.ncu:
        ncu_profile_all(args.versions, m_size, n_size, k_size)
    else:
        benchmark_all(args.versions, m_size, n_size, k_size)


if __name__ == "__main__":
    main()
