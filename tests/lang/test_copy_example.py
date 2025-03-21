import tilus
import torch
from tilus import float32, int32
from tilus.ir.layout import RegisterLayout, spatial
from tilus.utils import cdiv


class MemoryCopy(tilus.Script):
    def __init__(self, num_warps: int = 4):
        super().__init__()
        self.num_warps: int = num_warps
        self.block_size: int = num_warps * 32
        self.layout: RegisterLayout = spatial(num_warps * 32)

    def __call__(self, n: int32, src_ptr: ~float32, dst_ptr: ~float32):  # type: ignore
        self.attrs.blocks = [cdiv(n, self.block_size) * self.block_size]
        self.attrs.warps = self.num_warps

        bi = self.blockIdx.x

        loaded_regs = self.load_global_generic(
            dtype=float32,
            layout=self.layout,
            ptr=src_ptr,
            f_offset=lambda i: bi * self.block_size + i,
            f_mask=lambda i: bi * self.block_size + i < n,
        )
        self.store_global_generic(
            loaded_regs,
            ptr=dst_ptr,
            f_offset=lambda i: bi * self.block_size + i,
            f_mask=lambda i: bi * self.block_size + i < n,
        )


def test_tilus_script_with_copy_example():
    a = torch.ones(12, dtype=torch.float32).cuda()
    b = torch.empty(12, dtype=torch.float32).cuda()

    script = MemoryCopy()
    n = a.size(0)

    # launch the kernel
    script(n, a, b)

    torch.testing.assert_close(a, b)
