import torch
import tilus
from tilus import float16, int32, uint64
from tilus.utils import cdiv

tilus.option.cache_dir('./cache')
tilus.option.debug.dump_ir()
tilus.target.set_current_target(tilus.target.nvgpu_sm90a)

class BulkCopyAsyncExample(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_m = 64
        self.block_n = 32

    def __call__(self, m_size: int32, n_size: int, x_ptr: ~float16, y_ptr: ~float16):
        self.attrs.blocks = cdiv(m_size, self.block_m), cdiv(n_size, self.block_n)
        self.attrs.warps = 4

        m_offset = self.blockIdx.x * self.block_m
        n_offset = self.blockIdx.y * self.block_n

        g_x = self.global_view(x_ptr, dtype=float16, shape=[self.block_m, self.block_n])
        g_y = self.global_view(y_ptr, dtype=float16, shape=[self.block_m, self.block_n])

        s_x = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_n])
        s_y = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_n])

        barriers = self.shared_tensor(dtype=uint64, shape=[1])

        load_barrier: ~uint64 = ~barriers[0]
        self.init_barrier(load_barrier)
        self.bulk_copy_async_global_to_shared(
            src=g_x,
            dst=s_x,
            offsets=[m_offset, n_offset],
            mbarrier=load_barrier,
        )
        self.arrive_barrier(load_barrier)
        self.wait_barrier(load_barrier, phase=0)

        x = self.load_shared(s_x)
        x += 1
        self.store_shared(s_y, x)
        self.sync()

        self.bulk_copy_async_shared_to_global(
            src=s_y,
            dst=g_y,
            offsets=[m_offset, n_offset],
        )
        self.copy_async_wait_all()


def main():
    m = 1231
    n = 512
    x = torch.randn(m, n, dtype=torch.float16, device="cuda")
    y = torch.zeros_like(x)
    kernel = BulkCopyAsyncExample()
    kernel(m, n, x, y)

    torch.testing.assert_close(y, y + 1)


if __name__ == "__main__":
    main()
