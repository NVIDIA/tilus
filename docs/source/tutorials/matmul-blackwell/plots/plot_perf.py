"""Shared performance plotting for Blackwell matmul tutorials.

Benchmark data collected on B200 via NCU profiling (M=N=K=8192, fp16).
"""

import matplotlib.pyplot as plt
import numpy as np

# --- Benchmark data (B200, M=N=K=8192, fp16, NCU profiling) ---

VERSIONS = ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]

# Optimization label for each version
LABELS = [
    "MMA",
    "TMA",
    "Pipelining",
    "Warp Specialization",
    "Tile Rasterization",
    "Persistent Kernel",
    "2-CTA MMA",
]

# TFLOPS from NCU profiling
TFLOPS = [490.9, 454.3, 1103.2, 1351.1, 1416.4, 1506.3, 1610.2]

# NCU tensor core utilization (used to derive peak)
CUBLAS_TFLOPS = 1599.8
CUBLAS_TENSOR_UTIL = 96.6

# Peak TFLOPS (estimated: cuBLAS_TFLOPS / tensor_core_util)
PEAK_TFLOPS = CUBLAS_TFLOPS / (CUBLAS_TENSOR_UTIL / 100.0)


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
    ax.set_xlim(-0.3, n_total - 0.5 + 0.35)
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

        # Optimization label below the point
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
