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
from tilus.utils import cdiv


class CopyAsyncTensorExample(tilus.Script):
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
        self.init_barrier(load_barrier, count=1)
        self.sync()

        with self.single_thread():
            self.copy_async_tensor_global_to_shared(
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

        self.fence_proxy_copy_async()
        self.sync()

        with self.single_thread():
            self.copy_async_tensor_shared_to_global(
                src=s_y,
                dst=g_y,
                offsets=[m_offset, n_offset],
            )
            self.copy_async_tensor_commit_group()
            self.copy_async_tensor_wait_group(0)


def test_copy_async_tensor_cta():
    m = 123
    n = 64 * 8
    x = torch.randn(m, n, dtype=torch.float16, device="cuda")
    y = torch.zeros(m, n, dtype=torch.float16, device="cuda")
    kernel = CopyAsyncTensorExample()
    kernel(m, n, x, y)

    torch.cuda.synchronize()

    torch.testing.assert_close(y, x + 1)
