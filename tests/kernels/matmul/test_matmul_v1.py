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


class MatmulV1(tilus.Script):
    def __init__(self):
        super().__init__()
        self.mma = self.cuda.resolve_dot_config(float16, float32, num_warps=1, m=16, n=8, k=16)
        self.block_m = self.mma.m
        self.block_n = self.mma.n
        self.block_k = self.mma.k

    def __call__(self, m_size: int32, n_size: int, k_size: int, a_ptr: ~float16, b_ptr: ~float16, c_ptr: ~float16):
        self.attrs.blocks = [self.utils.ceil_div(m_size, self.block_m), self.utils.ceil_div(n_size, self.block_n)]
        self.attrs.warps = 1

        offset_m: int32 = self.block_m * self.blockIdx.x
        offset_n: int32 = self.block_n * self.blockIdx.y

        ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])
        sa = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_k])
        sb = self.shared_tensor(dtype=float16, shape=[self.block_k, self.block_n])
        acc = self.register_tensor(dtype=float32, shape=[self.block_m, self.block_n], init=0.0)

        for offset_k in range(0, k_size, self.block_k):
            lda = self.load_global(ga, offsets=[offset_m, offset_k], shape=[self.block_m, self.block_k])
            self.store_shared(sa, lda)
            ldb = self.load_global(gb, offsets=[offset_k, offset_n], shape=[self.block_k, self.block_n])
            self.store_shared(sb, ldb)
            self.sync()

            a = self.load_shared(sa, layout=self.mma.la)
            b = self.load_shared(sb, layout=self.mma.lb)
            acc = self.dot(a, b, acc)
            self.sync()

        self.free_shared(sa)
        self.free_shared(sb)

        casted_acc = self.cast(acc, dtype=float16)
        gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(gc, casted_acc, offsets=[offset_m, offset_n])


@pytest.mark.parametrize("m", [129, 257, 511])
@pytest.mark.parametrize("n, k", [[234, 456]])
def test_matmul_v1(m, n, k):
    matmul = MatmulV1()
    a = (torch.rand(m, k, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    b = (torch.rand(k, n, dtype=torch.float16).cuda() - 0.5) / math.sqrt(k)
    c = torch.empty(m, n, dtype=torch.float16).cuda()
    c_ref = a @ b

    matmul(m, n, k, a, b, c)

    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual=c,
        expected=c_ref,
    )
