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
import shutil
import subprocess
import time

VERSION_NAMES = ["v0", "v1", "v2", "v3", "v4", "v5", "v6"]

VERSION_CLASS = {
    "v0": "MatmulTMA",
    "v1": "MatmulWGMMA",
    "v2": "MatmulWGMMAV2",
    "v3": "MatmulWGMMAV3",
    "v4": "MatmulWGMMAV4",
    "v5": "MatmulWGMMAV5",
    "v6": "MatmulWGMMAV6",
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
    ncu = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    cmd = [ncu, "--import", report_path, "--csv", "--page", page]
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


# Timing protocol, following examples/blackwell_matmul/benchmark.py: CUDA-event
# timing via benchmark_func (median of `REPEAT` iterations, L2 flushed before
# each), and a `COOLDOWN_S` pause before every measurement -- including cuBLAS,
# which is timed last like any other entry -- so nothing is measured at a
# thermal state the rest did not see.
#
# REPEAT deliberately differs from the Blackwell script's 100. An 8192^3 fp16
# GEMM takes ~1.5 ms on an H100, so 100 back-to-back iterations is ~150 ms of
# power-capped tensor core work and the board throttles partway through the
# timed window. Measured over three fresh processes at this shape, repeat=30
# gives v6/cuBLAS = 108.1 / 106.5 / 107.8 %, while repeat=100 gives
# 99.0 / 98.9 / 110.4 % -- same kernels, but the median lands on whichever side
# of the throttle transition the run happened to sit. Pass --repeat to compare.
WARMUP = 5
REPEAT = 30
COOLDOWN_S = 3

# v6 accumulates in fp16 inside the WGMMA (see matmul_v6.py); every other version
# and cuBLAS accumulate in fp32. Measured at 8192^3 with unit-variance outputs,
# that raises the mean absolute error from 1.4e-4 to 2.3e-3, so v6 needs its own
# tolerance. This is a real precision difference, not a benchmarking artifact --
# see the warning in docs/source/tutorials/matmul-hopper/v6.rst.
FP16_ACCUMULATE_VERSIONS = {"v6"}


def benchmark_all(
    versions: list[str],
    m_size: int,
    n_size: int,
    k_size: int,
    repeat: int = REPEAT,
):
    """Benchmark all versions and cuBLAS using benchmark_func (CUDA-event timing)."""
    import pandas
    import torch
    from tilus.utils import benchmark_func

    headers = ["version", "accumulate", "latency (ms)", "tflops", "% of cublas"]
    rows = []

    # Scale both operands by k**0.25 so that C has unit variance: each output
    # element sums k products of two N(0, k**-0.5) values. That keeps atol and
    # rtol comparable in the correctness check below -- with unscaled operands C
    # has standard deviation sqrt(k), and with both operands scaled by 1/sqrt(k)
    # it has 1/sqrt(k), which makes an absolute tolerance meaningless in either
    # direction.
    scale = k_size**0.25
    a = torch.randn(m_size, k_size, dtype=torch.float16, device="cuda") / scale
    b = torch.randn(n_size, k_size, dtype=torch.float16, device="cuda") / scale
    c_ref = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")
    c_tilus = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")

    def tf(ms):
        return 2 * m_size * n_size * k_size / ms * 1e-9

    for name in versions:
        acc_dtype = "fp16" if name in FP16_ACCUMULATE_VERSIONS else "fp32"
        try:
            matmul = _load_version(name)()
            matmul(m_size, n_size, k_size, a, b, c_tilus)
            torch.cuda.synchronize()
            torch.matmul(a, b.T, out=c_ref)
            torch.cuda.synchronize()
            atol = 5e-2 if acc_dtype == "fp16" else 1e-2
            torch.testing.assert_close(c_tilus, c_ref, atol=atol, rtol=1e-2)

            time.sleep(COOLDOWN_S)
            t = benchmark_func(
                lambda: matmul(m_size, n_size, k_size, a, b, c_tilus),
                warmup=WARMUP,
                repeat=repeat,
            )
            rows.append([f"tilus_{name}", acc_dtype, t, tf(t), float("nan")])
        except Exception as e:
            print(f"  tilus_{name}  ERROR: {type(e).__name__}: {e}")
            rows.append(
                [f"tilus_{name}", acc_dtype, float("nan"), float("nan"), float("nan")]
            )

    # cuBLAS last, under the same cooldown and iteration count as every version.
    time.sleep(COOLDOWN_S)
    cublas_lat = benchmark_func(
        lambda: torch.matmul(a, b.T, out=c_ref), warmup=WARMUP, repeat=repeat
    )
    cublas_tf = tf(cublas_lat)
    rows.append(["cublas", "fp32", cublas_lat, cublas_tf, 100.0])

    for row in rows[:-1]:
        row[4] = row[3] / cublas_tf * 100.0

    df = pandas.DataFrame(rows, columns=headers)
    print(
        f"\nBenchmark results (m={m_size}, n={n_size}, k={k_size}, "
        f"warmup={WARMUP}, repeat={repeat}):"
    )
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
    parser = argparse.ArgumentParser(description="Benchmark Hopper matmul V0-V6")
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
    parser.add_argument(
        "--repeat",
        type=int,
        default=REPEAT,
        help=f"Timed iterations per measurement (default: {REPEAT})",
    )
    args = parser.parse_args()
    m_size, n_size, k_size = args.size

    if args.ncu:
        ncu_profile_all(args.versions, m_size, n_size, k_size)
    else:
        benchmark_all(args.versions, m_size, n_size, k_size, repeat=args.repeat)


if __name__ == "__main__":
    main()
