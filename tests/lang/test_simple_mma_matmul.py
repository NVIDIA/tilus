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
import math

import pytest
import tilus
import torch
from tilus import float16, float32, int32


class Matmul(tilus.Script):
    def __init__(self, n_size: int, k_size: int):
        super().__init__()
        self.mma = self.cuda.atomic_mma_configs.m16n8k16_f16_f32
        self.n_size = n_size
        self.k_size = k_size
        self.block_m = self.mma.m
        self.block_n = self.mma.n
        self.block_k = self.mma.k

    def __call__(self, m_size: int32, a_ptr: ~float16, b_ptr: ~float16, c_ptr: ~float16):
        self.attrs.blocks = [self.utils.ceil_div(m_size, self.block_m), self.utils.ceil_div(self.n_size, self.block_n)]
        self.attrs.warps = 1

        offset_m: int32 = self.block_m * self.blockIdx.x
        offset_n: int32 = self.block_n * self.blockIdx.y

        acc = self.register_tensor(dtype=float32, shape=self.mma.lc.shape, init=lambda indices: float32.zero)
        k_blocks = self.utils.ceil_div(self.k_size, self.block_k)
        for k in range(k_blocks):
            offset_k = k * self.block_k
            a = self.load_global_generic(
                dtype=float16,
                shape=self.mma.la.shape,
                ptr=a_ptr,
                f_offset=lambda i, k: (offset_m + i) * self.k_size + offset_k + k,
                f_mask=lambda i, k: offset_m + i < m_size and offset_k + k < self.k_size,
            )
            b = self.load_global_generic(
                dtype=float16,
                shape=self.mma.lb.shape,
                ptr=b_ptr,
                f_offset=lambda k, j: (offset_k + k) * self.n_size + offset_n + j,
                f_mask=lambda k, j: offset_k + k < self.k_size and offset_n + j < self.n_size,
            )
            acc = self.dot(a, b, acc)
        acc_f16 = self.cast(acc, dtype=float16)
        self.store_global_generic(
            acc_f16,
            ptr=c_ptr,
            f_offset=lambda i, j: (offset_m + i) * self.n_size + offset_n + j,
            f_mask=lambda i, j: offset_m + i < m_size and offset_n + j < self.n_size,
        )


@pytest.mark.parametrize("m,n,k", [[131, 137, 139], [345, 456, 567]])
def test_simple_mma_matmul(m, n, k):
    matmul = Matmul(n, k)
    a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    c = torch.empty(m, n, dtype=torch.float16).cuda()
    c_ref = a @ b
    matmul(m, a, b, c)

    torch.testing.assert_close(
        actual=c,
        expected=c_ref,
    )
