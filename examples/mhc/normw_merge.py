# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""mHC norm-weight merge (forward).

This is the load-bearing elementwise mHC primitive: ``out = fn * normw``.
"""

import tilus
import torch
from tile_kernels.mhc.norm_fn_kernel import _mhc_fn_normw_merge_fwd
from tilus import float32, int32
from tilus.utils import benchmark_func, cdiv


class MHCNormWMerge(tilus.Script):
    # One 256-wide row per CTA and two values per lane. Coalescing rows leaves
    # this small mHC launch with too few CTAs, while 256 threads makes each
    # lane handle only one value.
    def __init__(self, block_m: int = 1, block_n: int = 256, warps: int = 4):
        super().__init__()
        self.block_m, self.block_n, self.warps = block_m, block_n, warps

    def __call__(
        self, m: int32, n: int32, fn_ptr: ~float32, normw_ptr: ~float32, out_ptr: ~float32
    ):
        self.attrs.blocks = (cdiv(m, self.block_m), cdiv(n, self.block_n))
        self.attrs.warps = self.warps
        fn = self.global_view(fn_ptr, dtype=float32, shape=[m, n])
        normw = self.global_view(normw_ptr, dtype=float32, shape=[n])
        out = self.global_view(out_ptr, dtype=float32, shape=[m, n])
        rows = self.blockIdx.x * self.block_m
        cols = self.blockIdx.y * self.block_n
        r_fn = self.load_global(
            fn, offsets=[rows, cols], shape=[self.block_m, self.block_n]
        )
        r_w = self.load_global(normw, offsets=[cols], shape=[self.block_n])
        self.store_global(out, r_fn * r_w, offsets=[rows, cols])


def main():
    # These are the mHC shapes used by the TileKernels norm-fn path: fn is
    # [mhc_mult ** 3, mhc_hidden_size], normw is [mhc_hidden_size].
    m, n = 24, 4096
    fn = torch.randn((m, n), device="cuda", dtype=torch.float32)
    normw = torch.randn((n,), device="cuda", dtype=torch.float32)
    out = torch.empty_like(fn)
    kernel = MHCNormWMerge()
    kernel(m, n, fn, normw, out)
    tile_out = torch.empty_like(fn)
    tile_kernel = _mhc_fn_normw_merge_fwd(m, n)
    tile_kernel(fn, normw, tile_out)
    torch.testing.assert_close(out, fn * normw, rtol=0, atol=0)
    torch.testing.assert_close(out, tile_out, rtol=0, atol=0)

    def run_tilus():
        kernel(m, n, fn, normw, out)

    def run_tilekernels():
        tile_kernel(fn, normw, tile_out)

    tilus_ms = benchmark_func(run_tilus, warmup=10, repeat=100)
    tilekernels_ms = benchmark_func(run_tilekernels, warmup=10, repeat=100)
    print(
        f"mHC normw merge: Tilus {tilus_ms:.4f} ms, TileKernels {tilekernels_ms:.4f} ms"
    )


if __name__ == "__main__":
    main()
