import tilus
import torch
from tilus import int32, uint64


class DemoBarrier(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_size = 128

    def __call__(self, n: int32, x_ptr: ~int32, y_ptr: ~int32):
        self.attrs.blocks = 1
        self.attrs.warps = 2

        g_x = self.global_view(x_ptr, dtype=int32, shape=[n])
        g_y = self.global_view(y_ptr, dtype=int32, shape=[n])
        s_x = self.shared_tensor(dtype=int32, shape=[self.block_size])
        barriers = self.shared_tensor(dtype=uint64, shape=[1])

        self.init_barrier(~barriers[0], count=self.attrs.warps * 32)
        self.sync()

        phase: int32 = 0
        for bi in self.range(0, n, self.block_size):
            self.store_shared(
                dst=s_x, src=self.load_global(g_x, offsets=[bi * self.block_size], shape=[self.block_size])
            )

            self.arrive_barrier(~barriers[0])
            self.wait_barrier(~barriers[0], phase)
            phase ^= 1

            self.store_global(dst=g_y, src=self.load_shared(s_x) + 1, offsets=[bi * self.block_size])

            self.arrive_barrier(~barriers[0])
            self.wait_barrier(~barriers[0], phase)
            phase ^= 1


def test_mbarrier():
    n = 128
    x = torch.arange(n, dtype=torch.int32).cuda()
    y = torch.zeros(n, dtype=torch.int32).cuda()
    kernel = DemoBarrier()
    kernel(n, x, y)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x + 1)
