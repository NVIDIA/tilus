# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tilus
import torch
from tilus import float16, int32, uint64
from tilus.testing import requires
from tilus.utils import cdiv


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
        self.mbarrier.init(load_barrier)
        self.sync()

        self.tma.copy_async_bulk_global_to_shared(
            src=g_x,
            dst=s_x,
            offsets=[m_offset, n_offset],
            mbarrier=load_barrier,
        )
        self.mbarrier.arrive(load_barrier)
        self.mbarrier.wait(load_barrier, phase=0)

        x = self.load_shared(s_x)
        x += 1
        self.store_shared(s_y, x)
        self.tma.fence_proxy_copy_async()
        self.sync()

        self.tma.copy_async_bulk_shared_to_global(
            src=s_y,
            dst=g_y,
            offsets=[m_offset, n_offset],
        )
        self.tma.copy_async_tensor_commit_group()
        self.tma.copy_async_tensor_wait_group(0)


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
        self.mbarrier.init(load_barrier)
        self.cluster_sync()

        self.tma.copy_async_bulk_global_to_cluster_shared(  # type: ignore
            src=g_x, dst=s_x, offsets=[m_offset, n_offset], mbarrier=load_barrier, cta_mask=(1 << bs) - 1
        )
        self.mbarrier.arrive(load_barrier)
        self.mbarrier.wait(load_barrier, phase=0)

        x = self.load_shared(s_x)
        x += self.block_rank_in_cluster + 1
        self.store_shared(s_y, x)
        self.tma.fence_proxy_copy_async()
        self.sync()

        self.tma.copy_async_bulk_shared_to_global(  # type: ignore
            src=s_y, dst=g_y, offsets=[self.blockIdx.z, m_offset, n_offset], dims=[1, 2]
        )
        self.tma.copy_async_tensor_commit_group()
        self.tma.copy_async_tensor_wait_group(0)


@requires.nvgpu_sm90
def test_copy_async_bulk_cta():
    m = 123
    n = 64 * 8
    x = torch.randn(m, n, dtype=torch.float16, device="cuda")
    y = torch.zeros(m, n, dtype=torch.float16, device="cuda")
    kernel = BulkCopyAsyncExample()
    kernel(m, n, x, y)

    torch.testing.assert_close(y, x + 1)


@requires.nvgpu_sm90
def test_copy_async_bulk_cluster():
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
