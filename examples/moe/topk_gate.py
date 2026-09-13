# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stable MoE top-k routing, ported from TileKernels' ``topk_gate``.

Each CTA owns one token.  Repeated max/min reductions retain TileKernels'
tie rule: when equal scores occur, the lower expert index wins.
"""

import tilus
import torch
from tile_kernels.moe.topk_gate_kernel import topk_gate
from tilus import float32, int32, int64
from tilus.utils import benchmark_func, cdiv


class TopKGate(tilus.Script):
    def __init__(self, num_experts: int, num_topk: int):
        super().__init__()
        assert 1 <= num_topk <= num_experts
        self.num_experts = num_experts
        self.num_topk = num_topk
        self.aligned_experts = cdiv(num_experts, 32) * 32

    def __call__(self, num_tokens: int32, scores_ptr: ~float32, output_ptr: ~int64):
        self.attrs.blocks = (num_tokens,)
        self.attrs.warps = 1

        scores = self.global_view(scores_ptr, dtype=float32, shape=[num_tokens, self.num_experts])
        output = self.global_view(output_ptr, dtype=int64, shape=[num_tokens, self.num_topk])
        token = self.blockIdx.x

        values = self.load_global(scores, offsets=[token, 0], shape=[1, self.aligned_experts])
        # TileKernels performs the stable reducer in int32 and widens only at
        # the required int64 output boundary.
        expert_ids = self.register_tensor(
            dtype=int32, shape=[1, self.aligned_experts], init=lambda _, j: j
        )
        padding_value = -3.402823466e38
        # Keep selection state separate from score values: neither -inf nor
        # any finite sentinel is safe to use as an in-band removed marker.
        if self.aligned_experts != self.num_experts:
            active = self.register_tensor(
                dtype=int32,
                shape=[1, self.aligned_experts],
                init=lambda _, j: j < self.num_experts,
            )
        else:
            active = self.register_tensor(dtype=int32, shape=[1, self.aligned_experts], init=lambda _i, _j: 1)
        # Out-of-bounds vector-load lanes are not necessarily initialized to a
        # value below every valid score.  Mask them before the first max.
        values = self.where(active != 0, x=values, y=padding_value)

        # ``num_topk`` is a specialization constant.  TileKernels unrolls
        # this selection loop; retain that property in the generated CUDA.
        for rank in self.range(0, self.num_topk, 1, unroll="all"):
            best_value = self.max(values, dim=1, keepdim=True)
            # ``min`` over matching candidates is the stable tie breaker.
            candidates = self.where(
                active != 0,
                x=self.where(values == best_value, x=expert_ids, y=int32.max_value),
                y=int32.max_value,
            )
            best_index = self.min(candidates, dim=1, keepdim=True)
            # A sentinel maximum may also match a valid finite score.  Fall
            # back only when no active candidate matches the maximum (the
            # remaining scores are -inf).  This uniform branch avoids a second
            # warp reduction in the normal finite-score path.
            if best_index[0, 0].item() == int32.max_value:
                best_index = self.min(
                    self.where(active != 0, x=expert_ids, y=int32.max_value), dim=1, keepdim=True
                )
            # The reduction result is replicated across the warp.  A direct
            # store from every lane creates 32 identical global writes; only
            # lane 0 owns the scalar output (as in TileKernels' shared-output
            # epilogue).
            if self.get_thread_binding() == 0:
                self.store_global(output, best_index.to(int64), offsets=[token, rank])
            active = self.where(expert_ids == best_index, x=0, y=active)
            values = self.where(expert_ids == best_index, x=padding_value, y=values)


def main():
    rows = []
    for num_tokens, num_experts, num_topk in [(128, 72, 6), (1024, 256, 8), (8192, 256, 8)]:
        scores = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)
        # Make ties observable: the implementation must choose the lower index.
        scores[:, 0] = scores[:, 1]
        kernel = TopKGate(num_experts, num_topk)
        output = torch.empty(num_tokens, num_topk, device="cuda", dtype=torch.int64)
        kernel(num_tokens, scores, output)
        expected = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :num_topk]
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(topk_gate(scores, num_topk), expected)
        # Padding must never win when valid experts contain -inf.
        if num_experts % 32:
            edge_scores = torch.full_like(scores, float("-inf"))
            kernel(num_tokens, edge_scores, output)
            edge_expected = torch.arange(num_topk, device="cuda", dtype=torch.int64)[None, :].expand(
                num_tokens, -1
            )
            torch.testing.assert_close(output, edge_expected)
            # The minimum finite score must beat -inf even though it equals
            # the padding sentinel.
            edge_scores[:, 1] = torch.finfo(torch.float32).min
            kernel(num_tokens, edge_scores, output)
            edge_expected = torch.sort(edge_scores, dim=1, descending=True, stable=True).indices[:, :num_topk]
            torch.testing.assert_close(output, edge_expected)

        def run_tilus():
            out = torch.empty(num_tokens, num_topk, device="cuda", dtype=torch.int64)
            kernel(num_tokens, scores, out)
            return out

        tilus_ms = benchmark_func(run_tilus)
        tilekernels_ms = benchmark_func(lambda: topk_gate(scores, num_topk))
        rows.append((num_tokens, num_experts, num_topk, tilus_ms, tilekernels_ms))
    for row in rows:
        print("tokens=%d experts=%d topk=%d: Tilus %.4f ms, TileKernels %.4f ms" % row)


if __name__ == "__main__":
    main()
