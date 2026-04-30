# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import io
import subprocess
import time

VERSION_NAMES = ["v0", "v1", "v2", "v3", "v4", "v5", "v6"]


def _load_version(name: str):
    """Lazily import a matmul module by version name and return the class."""
    import importlib

    import tilus

    tilus.option.cache_dir("./cache")

    module = importlib.import_module(f"matmul_{name}")
    cls_name = f"BlackwellMatmul{name.upper()}"
    return getattr(module, cls_name)


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
    # the first data row is a units row (e.g. "%", "ms") — skip it
    next(reader, None)
    return reader


def _short_kernel_name(name: str) -> str:
    """Strip parameter list from a kernel name, e.g. 'foo(int, float *)' -> 'foo'."""
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

    # get DRAM/SM throughput and duration from the details page (SOL section)
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
    import pandas
    import torch
    from tilus.utils import benchmark_func

    headers = ["version", "latency (ms)", "tflops"]
    rows = []

    a = torch.randn(m_size, k_size, dtype=torch.float16, device="cuda")
    b = torch.randn(n_size, k_size, dtype=torch.float16, device="cuda")
    c = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")

    for name in versions:
        matmul = _load_version(name)()
        # warm up / correctness check
        matmul(m_size, n_size, k_size, a, b, c)
        torch.cuda.synchronize()

        latency = benchmark_func(
            lambda: matmul(m_size, n_size, k_size, a, b, c), warmup=5, repeat=100
        )
        tflops = 2 * m_size * n_size * k_size / latency * 1e-9
        rows.append(["tilus_" + name, latency, tflops])
        time.sleep(3)  # sleep 3s to cool down the GPU between runs

    # torch baseline
    latency = benchmark_func(lambda: a @ b.T, warmup=5, repeat=100)
    tflops = 2 * m_size * n_size * k_size / latency * 1e-9
    rows.append(["torch", latency, tflops])

    df = pandas.DataFrame(rows, columns=headers)
    print(f"\nBenchmark results (m={m_size}, n={n_size}, k={k_size}):")
    print(df.to_string(index=False))


def ncu_profile_all(versions: list[str], m_size: int, n_size: int, k_size: int):
    """Profile all versions in a single ncu_run and extract key metrics."""
    import pandas
    import tilus

    # warm up: trigger JIT compilation and autotuning before profiling
    print("Warming up (JIT + autotuning)...")
    run_kernels(versions, m_size, n_size, k_size)

    # labels: each tilus version first, then cuBLAS
    labels = list(versions) + ["cublas"]

    print(f"Profiling cublas, {', '.join(versions)} ...")
    report = tilus.utils.ncu_run(
        run_kernels,
        versions,
        m_size,
        n_size,
        k_size,
        kernel_regex="tilus|nvjet|gemm|cublas",
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
    parser = argparse.ArgumentParser(description="Benchmark Blackwell matmul V0-V6")
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
        """
        version                                         kernel  duration (ms)      tflops  sm_freq (GHz)  sm_throughput (%)  dram_throughput (%)  tensor_core_util (%)
            v0               tilus_blackwell_matmul_v0_kernel        2.24000  490.853405           1.75              25.92                12.74             23.241828
            v1               tilus_blackwell_matmul_v1_kernel        2.42000  454.343648           1.77              23.45                11.76             21.024327
            v2               tilus_blackwell_matmul_v2_kernel        0.99667 1103.185235           1.58              64.19                28.60             57.553969
            v3               tilus_blackwell_matmul_v3_kernel        0.81376 1351.149759           1.41              82.03                34.93             79.347981
            v4               tilus_blackwell_matmul_v4_kernel        0.77629 1416.367115           1.40              84.51                19.86             83.190080
            v5               tilus_blackwell_matmul_v5_kernel        0.72995 1506.283482           1.32              95.23                20.76             93.989628
            v6               tilus_blackwell_matmul_v6_kernel        0.68285 1610.180315           1.38              96.12                20.46             96.064392
        cublas nvjet_sm100_hsh_128x256_64x6_2x1_2cta_v_bz_TNT        0.68730 1599.755024           1.37              96.61                21.47             96.554842
        """
        ncu_profile_all(args.versions, m_size, n_size, k_size)
    else:
        """
        Sample output on B200 for M=N=K=8192
             version  latency (ms)      tflops
            tilus_v0      2.194464  501.038816
            tilus_v1      2.382288  461.535984
            tilus_v2      1.002576 1096.686570
            tilus_v3      0.845872 1299.855780
            tilus_v4      0.797712 1378.331566
            tilus_v5      0.756768 1452.904516
            tilus_v6      0.710576 1547.352614
               torch      0.711616 1545.091256
        """
        benchmark_all(args.versions, m_size, n_size, k_size)


if __name__ == "__main__":
    main()
