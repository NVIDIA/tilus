import tilus
from tilus import float32
from tilus.testing import requires


class PrintTmemTensor(tilus.Script):
    def __call__(self):
        self.attrs.blocks = 1
        self.attrs.warps = 4

        t_a = self.tcgen05.alloc(dtype=float32, shape=[128, 32])
        self.print_tensor("t_a: ", t_a)


@requires.nvgpu_sm100a
def test_print_tmem_tensor():
    kernel = PrintTmemTensor()
    kernel()
