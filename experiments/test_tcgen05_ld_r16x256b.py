"""
Minimal test: load_global -> store_shared -> load_shared -> MMA (to put data in tmem)
              -> tcgen05.load -> cast fp32->fp16 -> store_shared -> TMA to global

This tests the full epilogue path that matmul_v9 uses.
Simpler version: just load_global -> tcgen05.store -> tcgen05.load -> cast -> store_shared -> TMA
"""

import os
import shutil
import torch
import tilus
from tilus import RegisterTensor, SharedTensor, float16, float32, int32

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache_r16x256b")


class Tcgen05RoundTrip(tilus.Script):
    """
    Round-trip test: load fp16 from global -> cast to fp32 -> store to tmem ->
    load from tmem -> cast to fp16 -> store to shared -> TMA to global
    """

    debug_schedule = dict(block_m=128, block_n=64)

    def __init__(self, block_m: int, block_n: int):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n

    def __call__(self, m_size: int, n_size: int, in_ptr: ~float16, out_ptr: ~float16):
        self.attrs.blocks = 1, 1
        self.attrs.warps = 4

        block_m = self.block_m
        block_n = self.block_n

        g_in = self.global_view(in_ptr, dtype=float16, shape=[m_size, n_size])
        g_out = self.global_view(out_ptr, dtype=float16, shape=[m_size, n_size])

        # Load from global to registers
        r_in = self.load_global(g_in, offsets=[0, 0], shape=[block_m, block_n])

        # Cast to fp32 and store to tmem
        t_acc = self.tcgen05.alloc(dtype=float32, shape=[block_m, block_n], cta_group=1)
        self.tcgen05.store(t_acc, r_in.to(float32))
        self.tcgen05.wait_store()

        # Load from tmem to registers (THIS is where R16x256B vs R32x32B matters)
        r_acc = self.tcgen05.load(t_acc)
        self.tcgen05.wait_load()

        # Cast back to fp16
        r_fp16 = r_acc.to(float16)

        # Store to shared memory (generic store, no stmatrix)
        s_out = self.shared_tensor(dtype=float16, shape=[block_m, block_n])
        self.store_shared(s_out, r_fp16)

        # Fence + sync + TMA to global
        self.fence.async_view(space="shared")
        self.sync()
        with self.single_thread():
            self.tma.shared_to_global(s_out, g_out, offsets=[0, 0], dims=[0, 1])
            self.tma.commit_group()
            self.tma.wait_group(n=0)

        self.sync()
        self.tcgen05.dealloc(t_acc)


def main():
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    tilus.option.cache_dir(CACHE_DIR)
    tilus.option.debug.dump_ir()

    kernel = Tcgen05RoundTrip()

    m, n = 128, 64
    # Create input with a known non-zero pattern
    inp = torch.arange(m * n, dtype=torch.float16, device="cuda").reshape(m, n)
    out = torch.empty(m, n, dtype=torch.float16, device="cuda")

    kernel(m, n, inp, out)
    torch.cuda.synchronize()

    print("Input (first 4x8):")
    print(inp[:4, :8])
    print("\nOutput (first 4x8):")
    print(out[:4, :8])

    if torch.allclose(out, inp, atol=1e-1, rtol=1e-1):
        print("\nPASSED")
    else:
        mismatch = (~torch.isclose(out, inp, atol=1e-1, rtol=1e-1)).sum().item()
        total = out.numel()
        print(f"\nFAILED: {mismatch}/{total} elements mismatch ({mismatch / total * 100:.1f}%)")
        diff = (out - inp).abs()
        print(f"Max abs diff: {diff.max().item()}")


if __name__ == "__main__":
    main()
