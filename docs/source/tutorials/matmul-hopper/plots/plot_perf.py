"""Shared performance plotting for Hopper matmul tutorials.

Benchmark data collected on an H100 80GB HBM3 (SXM) with
``examples/hopper_matmul/benchmark.py`` at M=N=K=8192, fp16.

Latencies are CUDA-event timings (5 warmup + 30 timed iterations), taken as the
median of three fresh processes. TFLOPS = 2*M*N*K / latency.

Note that V6 accumulates in fp16 while V0-V5 and cuBLAS accumulate in fp32; see
the tutorial's V6 page for what that costs in accuracy.
"""

import matplotlib.pyplot as plt
import numpy as np

# --- Benchmark data (H100 SXM, M=N=K=8192, fp16, CUDA-event timing) ---

VERSIONS = ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]

# Optimization label for each version
LABELS = [
    "TMA + MMA",
    "WGMMA",
    "Pipelining",
    "Warp Spec.",
    "2 Consumers",
    "WGMMA Overlap",
    "4 Consumers",
]

# Median latency in ms over three fresh processes
LATENCY_MS = [3.600, 2.037, 2.124, 1.913, 1.714, 1.623, 1.374]
CUBLAS_LATENCY_MS = 1.481

_FLOP = 2 * 8192**3


def _tflops(ms):
    return _FLOP / (ms * 1e-3) * 1e-12


TFLOPS = [_tflops(ms) for ms in LATENCY_MS]
CUBLAS_TFLOPS = _tflops(CUBLAS_LATENCY_MS)

# Published dense FP16 tensor core throughput of the H100 SXM.
PEAK_TFLOPS = 989.4


def plot_performance(up_to_version: int | None = None):
    """Plot TFLOPS for tutorial versions.

    Parameters
    ----------
    up_to_version : int or None
        If given, highlight V0 through V{up_to_version} (solid line) and
        show remaining versions as dashed (preview). Labels are only shown
        for the highlighted versions.
        If None, show all versions as solid with labels.
    """
    n_total = len(VERSIONS)
    if up_to_version is not None:
        n_solid = up_to_version + 1
    else:
        n_solid = n_total

    x = np.arange(n_total)

    fig, ax = plt.subplots(figsize=(max(6.0, 1.25 * n_total + 1.2), 4.2))

    # Solid line: current and past versions
    ax.plot(x[:n_solid], TFLOPS[:n_solid], "o-", color="#5B9BD5", linewidth=2.2, markersize=8, zorder=4)

    # Dashed line: future versions (preview)
    if n_solid < n_total:
        x_dash = x[n_solid - 1 :]
        y_dash = TFLOPS[n_solid - 1 :]
        ax.plot(x_dash, y_dash, "o--", color="#5B9BD5", linewidth=1.2, markersize=5, alpha=0.35, zorder=3)

    # cuBLAS reference line
    ax.axhline(y=CUBLAS_TFLOPS, color="#E07B39", linewidth=1.5, linestyle="--", zorder=2)

    # Peak TFLOPS reference line
    ax.axhline(y=PEAK_TFLOPS, color="#888888", linewidth=1.5, linestyle="--", zorder=2)

    # Inline labels for reference lines (left side, bold)
    ax.text(
        0.02,
        PEAK_TFLOPS + PEAK_TFLOPS * 0.012,
        f"Peak ({PEAK_TFLOPS:.0f} TFLOPS)",
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#888888",
        transform=ax.get_yaxis_transform(),
    )
    ax.text(
        0.02,
        CUBLAS_TFLOPS - PEAK_TFLOPS * 0.012,
        f"cuBLAS ({CUBLAS_TFLOPS:.0f} TFLOPS)",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color="#E07B39",
        transform=ax.get_yaxis_transform(),
    )

    ax.set_ylabel("TFLOPS", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(VERSIONS, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    # Extra room on the right so the final version's label stays inside the axes
    ax.set_xlim(-0.3, n_total - 0.5 + 0.85)
    ax.set_ylim(0, PEAK_TFLOPS * 1.22)

    # Labels near solid points
    for i in range(n_solid):
        yi = TFLOPS[i]

        # TFLOPS value above the point, with white background to avoid
        # overlap with reference lines
        ax.annotate(
            f"{yi:.0f}",
            xy=(x[i], yi),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#333",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            zorder=5,
        )

        # Optimization label below and to the right of the point
        ax.annotate(
            LABELS[i],
            xy=(x[i], yi),
            xytext=(4, -14),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9,
            color="#666",
            style="italic",
        )

    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()

    return fig
