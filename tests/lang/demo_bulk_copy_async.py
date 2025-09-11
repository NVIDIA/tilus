import tilus
import torch
from tilus import float16, int32, uint64
from tilus.utils import cdiv


tilus.option.cache_dir('./cache')
tilus.option.debug.dump_ir()
# tilus.utils.clear_cache()


class BulkCopyAsyncExample(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_m = 32
        self.block_n = 64

    def __call__(self, m_size: int32, n_size: int, x_ptr: ~float16, y_ptr: ~float16):
        self.attrs.blocks = cdiv(m_size, self.block_m), cdiv(n_size, self.block_n)
        self.attrs.warps = 4

        m_offset = self.blockIdx.x * self.block_m
        n_offset = self.blockIdx.y * self.block_n

        g_x = self.global_view(x_ptr, dtype=float16, shape=[m_size, n_size])
        g_y = self.global_view(y_ptr, dtype=float16, shape=[m_size, n_size])

        s_x = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_n])
        s_y = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_n])

        barriers = self.shared_tensor(dtype=uint64, shape=[1])

        load_barrier: ~uint64 = ~barriers[0]
        self.init_barrier(load_barrier)
        self.sync()

        self.copy_async_bulk_global_to_shared(
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

        self.copy_async_bulk_shared_to_global(
            src=s_y,
            dst=g_y,
            offsets=[m_offset, n_offset],
        )
        self.copy_async_wait_all()


class BulkCopyAsyncClusterExample(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_m = 32
        self.block_n = 64

    def __call__(self, m_size: int32, n_size: int, x_ptr: ~float16, y_ptr: ~float16):
        self.attrs.blocks = cdiv(m_size, self.block_m), cdiv(n_size, self.block_n), 2
        self.attrs.cluster_blocks = (1, 1, 2)
        self.attrs.warps = 4

        m_offset = self.blockIdx.x * self.block_m
        n_offset = self.blockIdx.y * self.block_n

        g_x = self.global_view(x_ptr, dtype=float16, shape=[m_size, n_size])
        g_y = self.global_view(y_ptr, dtype=float16, shape=[2, m_size, n_size])

        s_x = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_n])
        s_y = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_n])

        barriers = self.shared_tensor(dtype=uint64, shape=[1])

        load_barrier: ~uint64 = ~barriers[0]
        self.init_barrier(load_barrier)
        self.sync()

        self.copy_async_bulk_global_to_cluster_shared(
            src=g_x,
            dst=s_x,
            offsets=[m_offset, n_offset],
            mbarrier=load_barrier,
            cta_mask=0b11
        )
        self.arrive_barrier(load_barrier)
        self.wait_barrier(load_barrier, phase=0)

        x = self.load_shared(s_x)
        if self.block_rank_in_cluster == 0:
            x += 1
        else:
            x += 2
        self.store_shared(s_y, x)
        self.sync()

        self.copy_async_bulk_shared_to_global(
            src=s_y,
            dst=g_y,
            offsets=[self.blockIdx.z, m_offset, n_offset],
            dims=[1, 2]
        )
        self.copy_async_wait_all()


def demo_copy_async_bulk_cta():
    m = 123
    n = 64 * 8
    x = torch.randn(m, n, dtype=torch.float16, device="cuda")
    y = torch.zeros(m, n, dtype=torch.float16, device="cuda")
    kernel = BulkCopyAsyncExample()
    kernel(m, n, x, y)

    torch.testing.assert_close(y, x + 1)


def demo_copy_async_bulk_cluster():
    m = 123
    n = 64 * 8
    x = torch.randn(m, n, dtype=torch.float16, device="cuda")
    y = torch.zeros(2, m, n, dtype=torch.float16, device="cuda")
    kernel = BulkCopyAsyncClusterExample()
    kernel(m, n, x, y)

    expect = torch.stack([x + 1, x + 2], dim=0).reshape(2, m, n)
    actual = y

    torch.testing.assert_close(actual, expect)


if __name__ == '__main__':
    demo_copy_async_bulk_cluster()
