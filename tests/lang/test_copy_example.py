import torch

import tilus
from tilus import float32, int32
from tilus.ir.layout import Layout, spatial
from tilus.utils import cdiv


class MemoryCopy(tilus.Script):
    def __init__(self, num_warps: int = 4):
        super().__init__()
        self.num_warps: int = num_warps
        self.block_size: int = num_warps * 32
        self.layout: Layout = spatial(num_warps * 32)

    def kernel(self, n: int32, src_ptr: ~float32, dst_ptr: ~float32):
        self.attrs.blocks = [cdiv(n, self.block_size) * self.block_size]
        self.attrs.warps = self.num_warps

        bi = self.blockIdx.x

        loaded_regs = self.load_global(
            dtype=float32,
            layout=self.layout,
            ptr=src_ptr,
            f_offset=lambda indices: bi * self.block_size + indices[0],
            f_mask=lambda indices: bi * self.block_size + indices[0] < n
        )
        self.store_global(
            loaded_regs,
            ptr=dst_ptr,
            f_offset=lambda indices: bi * self.block_size + indices[0],
            f_mask=lambda indices: bi * self.block_size + indices[0] < n
        )


def test_tilus_script_with_copy_example():
    a = torch.ones(12, dtype=torch.float32).cuda()
    b = torch.empty(12, dtype=torch.float32).cuda()

    script = MemoryCopy()
    n = a.size(0)

    # get the program
    print(script.program())

    # get the compiled module and print the source
    print(script.compiled().source())

    # launch the kernel (when there is only one kernel defined)
    script(n, a, b)

    # launch by method
    script.kernel(n, a, b)

    torch.testing.assert_close(a, b)

