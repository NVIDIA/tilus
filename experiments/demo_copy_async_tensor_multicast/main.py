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
import os

import tilus
import torch
from tilus import float16, int32, float32
from tilus.testing import requires
from tilus.utils import cdiv

tilus.option.cache_dir(os.path.join(os.path.dirname(__file__), "./cache"))


class CopyAsyncTensor2dMulticastExample(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_m = 128
        self.block_n = 64
        self.block_k = 16

    def __call__(self, m_size: int, n_size: int, k_size: int, a_ptr: ~float16, b_ptr: ~float16, c_ptr: ~float16):
        self.attrs.blocks = cdiv(m_size, self.block_m * 2) * 2, cdiv(n_size, self.block_n * 2) * 2
        self.attrs.cluster_blocks = 2, 2
        self.attrs.warps = 4

        g_a = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        g_b = self.global_view(b_ptr, dtype=float16, shape=[n_size, k_size])

        s_a = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_k])
        s_b = self.shared_tensor(dtype=float16, shape=[self.block_n, self.block_k])

        load_barrier = self.mbarrier.alloc(count=4)
        self.cluster_sync()

        m_offset = self.blockIdx.x * self.block_m 
        n_offset = self.blockIdx.y * self.block_n

        s_a_reshaped = self.reshape_shared(s_a, [2, self.block_m // 2, self.block_k])
        s_b_reshaped = self.reshape_shared(s_b, [2, self.block_n // 2, self.block_k])

        mma_barrier = self.mbarrier.alloc(count=1)
        t_acc = self.tcgen05.alloc(dtype=float32, shape=[self.block_m, self.block_n], init=0.0)

        self.sync()

        phase: int32 = 0

        for k_offset in self.range(0, k_size, self.block_k):
            with self.single_warp():
                """
                    |  B0  |  B1  |  B2  |  B3  |
                ----+-------------+-------------+
                A0  |             |             |
                ----|     cta0    |     cta2    |
                A1  |             |             |
                ----+-------------+-------------+
                A2  |             |             |
                ----|     cta1    |     cta3    |
                A3  |             |             |
                ----+-------------+-------------+

                cta0: load A0 and B0
                cta1: load A2 and B1
                cta2: load A1 and B2
                cta3: load A3 and B3
                """
                self.tma.global_to_shared(
                    src=g_a,
                    dst=s_a_reshaped[self.cluster.blockIdx.y],
                    offsets=[m_offset + self.cluster.blockIdx.y * (self.block_m // 2), k_offset],
                    multicast_mask=0b0101 if self.cluster.blockIdx.x == 0 else 0b1010,
                    mbarrier=load_barrier,
                )
                self.tma.global_to_shared(
                    src=g_b,
                    dst=s_b_reshaped[self.cluster.blockIdx.x],
                    offsets=[n_offset + self.cluster.blockIdx.x * (self.block_n // 2), k_offset],
                    multicast_mask=0b0011 if self.cluster.blockIdx.y == 0 else 0b1100,
                    mbarrier=load_barrier,
                )

            self.mbarrier.wait(load_barrier, phase=phase)

            # mma
            with self.single_thread():
                self.tcgen05.mma(s_a, s_b.transpose(), t_acc)
                self.tcgen05.commit(mma_barrier)
            self.mbarrier.wait(mma_barrier, phase=phase)
            phase ^= 1

        r_acc = self.tcgen05.load(t_acc)
        self.tcgen05.wait_load()
        self.sync()
        self.tcgen05.dealloc(t_acc)

        # write back
        g_c = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(g_c, r_acc.to(float16), offsets=[m_offset, n_offset])

class CopyAsyncTensor2dPingpongMulticastExample(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_m = 128
        self.block_n = 64
        self.block_k = 16

    def __call__(self, m_size: int, n_size: int, k_size: int, a_ptr: ~float16, b_ptr: ~float16, c_ptr: ~float16):
        self.attrs.blocks = cdiv(m_size, self.block_m * 2) * 2, cdiv(n_size, self.block_n * 2) * 2
        self.attrs.cluster_blocks = 2, 2
        self.attrs.warps = 4

        g_a = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        g_b = self.global_view(b_ptr, dtype=float16, shape=[n_size, k_size])

        s_a = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_k])
        s_b = self.shared_tensor(dtype=float16, shape=[self.block_n, self.block_k])

        tma_barrier = self.mbarrier.alloc(count=4)
        mma_barrier = self.mbarrier.alloc(count=1)

        m_offset = self.blockIdx.x * self.block_m 
        n_offset = self.blockIdx.y * self.block_n

        s_a_reshaped = self.reshape_shared(s_a, [2, self.block_m // 2, self.block_k])
        s_b_reshaped = self.reshape_shared(s_b, [2, self.block_n // 2, self.block_k])
        t_acc = self.tcgen05.alloc(dtype=float32, shape=[self.block_m, self.block_n], init=0.0)

        self.cluster_sync()
        worker_phase: int32 = 0
        loader_phase: int32 = 1

        self.printf("[%d, %d][%d, %d] %d\n", self.blockIdx.x, self.blockIdx.y, self.cluster.blockIdx.x, self.cluster.blockIdx.y, self.cluster.blockRank)

        with self.thread_group(0, num_threads=32):
            # loader
            for k_offset in self.range(0, k_size, self.block_k):
                """
                    |  B0  |  B1  |  B2  |  B3  |
                ----+-------------+-------------+
                A0  |             |             |
                ----|     cta0    |     cta2    |
                A1  |             |             |
                ----+-------------+-------------+
                A2  |             |             |
                ----|     cta1    |     cta3    |
                A3  |             |             |
                ----+-------------+-------------+

                cta0: load A0 and B0
                cta1: load A2 and B1
                cta2: load A1 and B2
                cta3: load A3 and B3
                """
                # self.printf("[%d, %d][%d, %d][Loader] start working, k_offset=%d\n", self.blockIdx.x // self.cluster.clusterDim.x, self.blockIdx.y // self.cluster.clusterDim.y, self.cluster.blockIdx.x, self.cluster.blockIdx.y, k_offset // self.block_k)
                self.mbarrier.wait(mma_barrier, phase=loader_phase)
                self.tma.global_to_shared(
                    src=g_a,
                    dst=s_a_reshaped[self.cluster.blockIdx.y],
                    offsets=[m_offset + self.cluster.blockIdx.y * (self.block_m // 2), k_offset],
                    multicast_mask=0b0101 if self.cluster.blockIdx.x == 0 else 0b1010,
                    mbarrier=tma_barrier,
                )
                self.tma.global_to_shared(
                    src=g_b,
                    dst=s_b_reshaped[self.cluster.blockIdx.x],
                    offsets=[n_offset + self.cluster.blockIdx.x * (self.block_n // 2), k_offset],
                    multicast_mask=0b0011 if self.cluster.blockIdx.y == 0 else 0b1100,
                    mbarrier=tma_barrier,
                )
                loader_phase ^= 1
                # self.cluster_sync()
            self.mbarrier.wait(mma_barrier, phase=loader_phase)

        with self.thread_group(32, num_threads=32):
            # mma
            for k_offset in self.range(0, k_size, self.block_k):
                # self.sync()
                # self.printf("[%d, %d][%d, %d][   MMA] start working, k_offset=%d\n", self.blockIdx.x // self.cluster.clusterDim.x, self.blockIdx.y // self.cluster.clusterDim.y, self.cluster.blockIdx.x, self.cluster.blockIdx.y, k_offset // self.block_k)
                self.mbarrier.wait(tma_barrier, phase=worker_phase)
                with self.single_thread():
                    self.tcgen05.mma(s_a, s_b.transpose(), t_acc)
                    self.tcgen05.commit(mma_barrier)
                worker_phase ^= 1
                # self.cluster_sync()
        
        # with self.thread_group(64, num_threads=64):
        #     for k_offset in self.range(0, k_size, self.block_k):
        #         self.cluster_sync()
        
        self.sync()

        r_acc = self.tcgen05.load(t_acc)
        self.tcgen05.wait_load()
        self.sync()
        self.tcgen05.dealloc(t_acc)

        # write back
        g_c = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(g_c, r_acc.to(float16), offsets=[m_offset, n_offset])

@requires.nvgpu_sm90
def test_copy_async_tensor_2d_cluster():
    m = 128 * 2
    n = 64 * 2
    k = 16 * 13
    a = torch.randint(0, 2, size=(m, k), dtype=torch.float16, device='cuda')
    b = torch.randint(0, 2, size=(n, k), dtype=torch.float16, device='cuda')
    c = torch.zeros((m, n), dtype=torch.float16, device='cuda')
    kernel = CopyAsyncTensor2dMulticastExample()
    kernel(m, n, k, a, b, c)
    torch.cuda.synchronize()

    expected = a @ b.T
    torch.cuda.synchronize()
    torch.testing.assert_close(c, expected)

@requires.nvgpu_sm90
def test_copy_async_tensor_2d_pingpong_cluster():
    m = 128 * 4 
    n = 64 * 2 
    k = 16 * 200
    a = torch.randint(0, 2, size=(m, k), dtype=torch.float16, device='cuda')
    b = torch.randint(0, 2, size=(n, k), dtype=torch.float16, device='cuda')
    c = torch.zeros((m, n), dtype=torch.float16, device='cuda')
    kernel = CopyAsyncTensor2dPingpongMulticastExample()
    kernel(m, n, k, a, b, c)
    torch.cuda.synchronize()

    expected = a @ b.T
    torch.cuda.synchronize()
    torch.testing.assert_close(c, expected)

if __name__ == "__main__":
    # pytest.main([__file__])
    # test_copy_async_tensor_cluster()
    # test_copy_async_tensor_2d_cluster()
    test_copy_async_tensor_2d_pingpong_cluster()
