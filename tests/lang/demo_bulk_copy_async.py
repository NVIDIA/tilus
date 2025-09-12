import tilus
import torch
from tilus import float16, int32, uint64
from tilus.utils import cdiv
import pandas as pd
from tilus.utils.cuda_blocking_run import cuda_blocking_run
from hidet.utils.cuda_sanitizer import sanitizer_run


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
        self.block_m = 1
        self.block_n = 64

    def __call__(self, bs: int, m_size: int32, n_size: int, x_ptr: ~float16, y_ptr: ~float16):
        self.attrs.blocks = cdiv(m_size, self.block_m), cdiv(n_size, self.block_n), bs
        self.attrs.cluster_blocks = (1, 1, bs)
        self.attrs.warps = 4

        m_offset = self.blockIdx.x * self.block_m
        n_offset = self.blockIdx.y * self.block_n

        g_x = self.global_view(x_ptr, dtype=float16, shape=[m_size, n_size])
        g_y = self.global_view(y_ptr, dtype=float16, shape=[bs, m_size, n_size])

        s_x = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_n])
        s_y = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_n])

        barriers = self.shared_tensor(dtype=uint64, shape=[1])

        load_barrier: ~uint64 = ~barriers[0]
        self.init_barrier(load_barrier)
        self.cluster_sync()

        self.copy_async_bulk_global_to_cluster_shared(
            src=g_x,
            dst=s_x,
            offsets=[m_offset, n_offset],
            mbarrier=load_barrier,
            cta_mask=(1 << bs) - 1
        )
        self.arrive_barrier(load_barrier)
        self.wait_barrier(load_barrier, phase=0)

        x = self.load_shared(s_x)
        x += self.block_rank_in_cluster + 1
        self.store_shared(s_y, x)
        self.sync()

        self.copy_async_bulk_shared_to_global(
            src=s_y,
            dst=g_y,
            offsets=[self.blockIdx.z, m_offset, n_offset],
            dims=[1, 2]
        )
        self.copy_async_wait_all()


class BulkCopyAsyncSharedToClusterSharedExample(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_m = 32
        self.block_n = 64

    def __call__(self, m_size: int32, n_size: int, x_ptr: ~float16, y_ptr: ~float16):
        self.attrs.blocks = cdiv(m_size, self.block_m), cdiv(n_size, self.block_n), 2
        self.attrs.cluster_blocks = 1, 1, 2
        self.attrs.warps = 4

        m_offset = self.blockIdx.x * self.block_m
        n_offset = self.blockIdx.y * self.block_n

        g_x = self.global_view(x_ptr, dtype=float16, shape=[m_size, n_size])
        g_y = self.global_view(y_ptr, dtype=float16, shape=[m_size, n_size])

        s_x = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_n])
        barriers = self.shared_tensor(dtype=uint64, shape=[2])

        if self.blockIdx.z == 0:
            self.init_barrier(~barriers[0])
            self.init_barrier(~barriers[1])
        else:
            self.init_barrier(~barriers[1], count=1)
        self.cluster_sync()

        self.printf('[%d, %d, %d][%d] cluster sync done\n', self.blockIdx.x, self.blockIdx.y, self.blockIdx.z, self.block_rank_in_cluster)

        if self.blockIdx.z == 0:
            self.copy_async_bulk_global_to_shared(
                src=g_x,
                dst=s_x,
                offsets=[m_offset, n_offset],
                mbarrier=~barriers[0],
            )
            self.arrive_barrier(~barriers[0])
            self.wait_barrier(~barriers[0], phase=0)

            self.printf('[%d, %d, %d][%d] copy global to shared done\n', self.blockIdx.x, self.blockIdx.y, self.blockIdx.z, self.block_rank_in_cluster)

            self.copy_async_bulk_shared_to_cluster_shared(
                src=s_x,
                dst=s_x,
                mbarrier=~barriers[1]
            )
            self.arrive_barrier(~barriers[1])
            self.wait_barrier(~barriers[1], phase=0)

            self.printf('[%d, %d, %d][%d] copy shared to cluster shared done\n', self.blockIdx.x, self.blockIdx.y, self.blockIdx.z, self.block_rank_in_cluster)

            self.arrive_remote_barrier(~barriers[1], remote_block=0x1)

            self.printf('[%d, %d, %d][%d] arrive remote barrier done\n', self.blockIdx.x, self.blockIdx.y, self.blockIdx.z, self.block_rank_in_cluster)
        else:
            self.printf('[%d, %d, %d][%d] before wait remote barrier\n', self.blockIdx.x, self.blockIdx.y, self.blockIdx.z, self.block_rank_in_cluster)
            self.wait_barrier(~barriers[1], phase=0)
            self.printf('[%d, %d, %d][%d] after wait remote barrier\n', self.blockIdx.x, self.blockIdx.y, self.blockIdx.z, self.block_rank_in_cluster)
            self.copy_async_bulk_shared_to_global(
                src=s_x,
                dst=g_y,
                offsets=[m_offset, n_offset],
            )
            self.copy_async_wait_all()
            self.printf('[%d, %d, %d][%d] copy shared to global done\n', self.blockIdx.x, self.blockIdx.y, self.blockIdx.z, self.block_rank_in_cluster)


def demo_copy_async_bulk_cta():
    m = 123
    n = 64 * 8
    x = torch.randn(m, n, dtype=torch.float16, device="cuda")
    y = torch.zeros(m, n, dtype=torch.float16, device="cuda")
    kernel = BulkCopyAsyncExample()
    kernel(m, n, x, y)

    torch.testing.assert_close(y, x + 1)


def demo_copy_async_bulk_cluster():
    bs = 4
    m = 123
    n = 64 * 32
    x = torch.ones(m, n, dtype=torch.float16, device="cuda")
    y = torch.zeros(bs, m, n, dtype=torch.float16, device="cuda")
    kernel = BulkCopyAsyncClusterExample()
    kernel(bs, m, n, x, y)

    torch.cuda.synchronize()

    expect = torch.stack([x + i + 1 for i in range(bs)], dim=0).reshape(bs, m, n)
    actual = y

    torch.testing.assert_close(actual, expect)

def demo_copy_async_bulk_shared_to_cluster_shared():
    m = 32
    n = 64
    x = torch.randn(m, n, dtype=torch.float16, device="cuda")
    y = torch.zeros(m, n, dtype=torch.float16, device="cuda")
    kernel = BulkCopyAsyncSharedToClusterSharedExample()
    kernel(m, n, x, y)

    torch.testing.assert_close(actual=y, expected=x)

if __name__ == '__main__':
    # demo_copy_async_bulk_cluster()
    demo_copy_async_bulk_shared_to_cluster_shared()
