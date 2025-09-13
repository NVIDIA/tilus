import tilus
import torch
from tilus import float16, int32
from tilus.utils import cdiv


class ThreadGroupExample(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block: int = 64

    def __call__(self, n: int32, x_ptr: ~float16, y_ptr: ~float16):
        self.attrs.blocks = cdiv(n, self.block)
        self.attrs.warps = 4

        g_x = self.global_view(x_ptr, dtype=float16, shape=[n])
        g_y = self.global_view(y_ptr, dtype=float16, shape=[n])

        s_x = self.shared_tensor(dtype=float16, shape=[self.block])
        offset = self.blockIdx.x * self.block

        with self.thread_group(0, group_size=64):
            self.store_shared(s_x, self.load_global(g_x, offsets=[offset], shape=[self.block]))
            self.sync()
        self.sync()

        r_x = self.load_shared(s_x)
        r_x += 1
        self.store_shared(dst=s_x, src=r_x)
        self.sync()

        with self.thread_group(1, group_size=64):
            self.store_global(dst=g_y, src=self.load_shared(s_x), offsets=[offset])

        self.sync()


def test_thread_group():
    n = 1024
    x = torch.randn(n, dtype=torch.float16, device="cuda")
    y = torch.zeros(n, dtype=torch.float16, device="cuda")

    kernel = ThreadGroupExample()
    kernel(n, x, y)

    expect = x + 1
    actual = y
    torch.testing.assert_close(actual=actual, expected=expect)


if __name__ == "__main__":
    test_thread_group()
