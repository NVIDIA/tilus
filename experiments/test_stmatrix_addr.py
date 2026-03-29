"""
Direct test of stmatrix address mapping.

Uses a simple kernel with load_global -> store_shared (with stmatrix).
The input is an identity-like pattern where element (i, j) = i * N + j,
so we can trace exactly where each element ends up.

Compares stmatrix store vs generic store to find the mapping.
"""

import os
import shutil
import torch
import tilus
from tilus import float16, int32

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache_stmatrix_addr")


class StmatrixDirect(tilus.Script):
    """Load from global, store to shared with stmatrix, TMA to global."""

    debug_schedule = dict(block_m=8, block_n=8)

    def __init__(self, block_m: int, block_n: int):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n

    def __call__(self, m_size: int, n_size: int, in_ptr: ~float16, out_ptr: ~float16):
        self.attrs.blocks = 1, 1
        self.attrs.warps = 1  # Single warp to simplify

        block_m = self.block_m
        block_n = self.block_n

        g_in = self.global_view(in_ptr, dtype=float16, shape=[m_size, n_size])
        g_out = self.global_view(out_ptr, dtype=float16, shape=[m_size, n_size])

        r = self.load_global(g_in, offsets=[0, 0], shape=[block_m, block_n])

        s_out = self.shared_tensor(dtype=float16, shape=[block_m, block_n])
        self.store_shared(s_out, r)

        self.fence.async_view(space="shared")
        self.sync()
        with self.single_thread():
            self.tma.shared_to_global(s_out, g_out, offsets=[0, 0], dims=[0, 1])
            self.tma.commit_group()
            self.tma.wait_group(n=0)


def main():
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    tilus.option.cache_dir(CACHE_DIR)
    tilus.option.debug.dump_ir()

    kernel = StmatrixDirect()

    m, n = 8, 8
    inp = torch.arange(m * n, dtype=torch.float16, device="cuda").reshape(m, n)
    out = torch.empty(m, n, dtype=torch.float16, device="cuda")

    kernel(m, n, inp, out)
    torch.cuda.synchronize()

    print("Input:")
    print(inp)
    print("\nOutput:")
    print(out)

    if torch.allclose(out, inp, atol=1e-1):
        print("\nPASSED")
    else:
        print("\nFAILED")
        # Show which element ended up where
        for i in range(m):
            for j in range(n):
                val = out[i, j].item()
                expected = inp[i, j].item()
                if val != expected:
                    # Find where this value came from in input
                    src_idx = int(val)
                    src_i, src_j = src_idx // n, src_idx % n
                    print(f"  out[{i},{j}] = {val} (from input[{src_i},{src_j}]), expected {expected}")


if __name__ == "__main__":
    main()
