# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the DSV3.2 sparse-attention indexer.

Columns:
  - baseline     : PyTorch reference from ``baseline.py`` (slow, correctness oracle).
  - vllm         : the unfused decode pipeline that vLLM uses for DeepSeek V3.2 —
                   ``vllm.utils.deep_gemm.fp8_paged_mqa_logits`` (MQA logits;
                   requires DeepGEMM) followed by vLLM's own
                   ``torch.ops._C.top_k_per_row_decode`` / ``large_context_topk``.
                   Skipped gracefully when either piece is missing.
  - tilus-logits : Tilus M1 logits kernel (``tilus_kernel.py``). Timed in
                   isolation — the Tilus radix top-K (M4) will land later
                   and a combined ``tilus`` column will follow once wired.

Usage:
    python benchmark.py
    python benchmark.py --batch-sizes 1 4 32 --seq-lens 4096 32768
"""

import argparse
import time

import torch

try:
    import vllm  # noqa: F401  — loads torch.ops._C

    from vllm.utils.deep_gemm import (
        fp8_paged_mqa_logits,
        get_paged_mqa_logits_metadata,
        has_deep_gemm,
    )

    HAS_VLLM = True
    HAS_DEEP_GEMM = has_deep_gemm()
except ImportError:
    HAS_VLLM = False
    HAS_DEEP_GEMM = False


def _deep_gemm_arch_supported() -> bool:
    """DeepGEMM's paged-MQA kernel only supports Hopper (SM90) and datacenter
    Blackwell (SM100/SM103). Workstation/consumer Blackwell (SM120/SM121) uses
    a different tensor-core ISA and is unsupported."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major in (9, 10)


def make_test_data(batch_size: int, seq_len: int):
    """Build fixed-length test data matching vLLM's paged-MQA layout."""
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
    num_pages_per_seq = (seq_len + 63) // 64
    total_pages = batch_size * num_pages_per_seq
    max_num_pages = (num_pages_per_seq + 1) // 2 * 2  # kernel requires even count

    q = torch.randn(batch_size, 64, 128, dtype=torch.float32, device="cuda").to(
        torch.float8_e4m3fn
    )

    k_cache = torch.empty(total_pages, 64, 1, 132, dtype=torch.uint8, device="cuda")
    flat = k_cache.view(total_pages, -1)
    flat[:, : 64 * 128].view(torch.float8_e4m3fn).copy_(
        torch.randn(total_pages, 64 * 128, dtype=torch.float32, device="cuda").to(
            torch.float8_e4m3fn
        )
    )
    flat[:, 64 * 128 :].view(torch.float32).copy_(
        torch.randn(total_pages, 64, dtype=torch.float32, device="cuda").abs()
    )

    weights = torch.randn(batch_size, 64, dtype=torch.float32, device="cuda")
    block_table = torch.zeros(
        batch_size, max_num_pages, dtype=torch.int32, device="cuda"
    )
    for b in range(batch_size):
        block_table[b, :num_pages_per_seq] = torch.arange(
            b * num_pages_per_seq,
            (b + 1) * num_pages_per_seq,
            dtype=torch.int32,
            device="cuda",
        )

    return q, k_cache, weights, seq_lens, block_table


def _vllm_unfused_pipeline(q, k_cache, weights, seq_lens, block_table, max_model_len):
    """Reproduce vLLM's DeepSeek V3.2 decode indexer pipeline.

    Mirrors ``vllm/model_executor/layers/sparse_attn_indexer.py`` decode path:
    1. ``fp8_paged_mqa_logits`` → ``logits[B*next_n, max_model_len]``
    2. ``top_k_per_row_decode`` or ``large_context_topk`` per vLLM's own heuristic.

    Returns a callable that runs the full pipeline, so ``benchmark_func`` can
    time it.
    """
    batch_size = q.shape[0]
    next_n = 1  # single decode token
    num_rows = batch_size * next_n

    # next_n=1 → shape [B, 1, 64, 128] for the API
    q_dg = q.view(batch_size, next_n, 64, 128).contiguous()

    # DeepGEMM v2.4+ requires 2D context_lens of shape [batch, next_n].
    seq_lens_2d = seq_lens.view(batch_size, next_n).contiguous()

    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    schedule_meta = get_paged_mqa_logits_metadata(seq_lens_2d, 64, num_sms)

    # Pre-allocate outputs (vLLM does the same; the kernels write in place).
    logits = torch.empty(
        num_rows, max_model_len, dtype=torch.float32, device=q.device
    )
    topk_indices = torch.empty(num_rows, 2048, dtype=torch.int32, device=q.device)

    max_seq_len = int(seq_lens.max().item())
    use_large_context = (batch_size <= 128) and (max_seq_len > 8192)

    def run():
        nonlocal logits
        logits = fp8_paged_mqa_logits(
            q_dg,
            k_cache,
            weights,
            seq_lens_2d,
            block_table,
            schedule_meta,
            max_model_len=max_model_len,
            clean_logits=False,
        )
        if use_large_context:
            torch.ops._C.large_context_topk(logits, topk_indices, seq_lens, None)
        else:
            torch.ops._C.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                2048,
            )

    return run


def benchmark(batch_sizes: list, seq_lens: list):
    import pandas
    from tilus.utils import benchmark_func

    from baseline import dsv3_topk_indexer_ref
    from tilus_kernel import dsv3_indexer_logits_tilus

    run_vllm_column = HAS_VLLM and HAS_DEEP_GEMM and _deep_gemm_arch_supported()

    rows = []
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            q, k_cache, w, sl, bt = make_test_data(batch_size, seq_len)
            max_model_len = bt.shape[1] * 64

            # PyTorch reference (correctness oracle; slow).
            baseline_ms = benchmark_func(
                lambda: dsv3_topk_indexer_ref(q, k_cache, w, sl, bt),
                warmup=2,
                repeat=5,
            )
            row = {
                "batch": batch_size,
                "seq_len": seq_len,
                "baseline (ms)": baseline_ms,
            }

            # vLLM's unfused decode pipeline.
            if run_vllm_column:
                run_vllm = _vllm_unfused_pipeline(
                    q, k_cache, w, sl, bt, max_model_len
                )
                vllm_ms = benchmark_func(run_vllm, warmup=5, repeat=50)
                row["vllm (ms)"] = vllm_ms

            # Tilus M1 logits kernel. Warm up once to pay the autotune +
            # JIT cost before timing.
            dsv3_indexer_logits_tilus(q, k_cache, w, sl, bt, max_model_len)
            tilus_ms = benchmark_func(
                lambda: dsv3_indexer_logits_tilus(
                    q, k_cache, w, sl, bt, max_model_len
                ),
                warmup=5,
                repeat=50,
            )
            row["tilus-logits (ms)"] = tilus_ms

            rows.append(row)
            time.sleep(1)  # GPU cool-down between runs

    df = pandas.DataFrame(rows)
    print("\nBenchmark results:")
    if not HAS_VLLM:
        print("(vllm not installed — only baseline is timed)")
    elif not HAS_DEEP_GEMM:
        print(
            "(vllm installed but DeepGEMM not available — pip install deep_gemm from "
            "https://github.com/deepseek-ai/DeepGEMM to enable the vllm column)"
        )
    elif not _deep_gemm_arch_supported():
        major, minor = torch.cuda.get_device_capability(0)
        print(
            f"(DeepGEMM does not support SM{major}{minor} — "
            f"fp8_paged_mqa_logits is gated to Hopper SM90 and datacenter Blackwell "
            f"SM100/SM103 only. vllm column is skipped on this GPU.)"
        )
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DSV3.2 sparse-attention indexer"
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 32, 64], metavar="B"
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[4096, 16384, 65536],
        metavar="L",
    )
    args = parser.parse_args()

    benchmark(args.batch_sizes, args.seq_lens)


if __name__ == "__main__":
    main()
