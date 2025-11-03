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
import pytest
import tilus
import torch
from hidet.ir.type import DataType
from tilus import float8_e4m3, float16, float32, void_p
from tilus.utils import cdiv, dtype_to_torch


class Tcgen05MmaExample(tilus.Script):
    def __init__(
        self,
        operand_dtype: DataType,
        accumulator_dtype: DataType,
        output_dtype: DataType,
        mma_m: int,
        mma_n: int,
        mma_k: int,
    ):
        super().__init__()
        self.operand_dtype = operand_dtype
        self.accumulator_dtype = accumulator_dtype
        self.output_dtype = output_dtype
        self.mma_m = mma_m
        self.mma_n = mma_n
        self.mma_k = mma_k
        self.column_granularity = 32 * 32 // self.accumulator_dtype.nbits

    def __call__(self, a_ptr: void_p, b_ptr: void_p, d_ptr: void_p):
        self.attrs.blocks = 1
        self.attrs.warps = 4

        g_a = self.global_view(a_ptr, dtype=self.operand_dtype, shape=[self.mma_m, self.mma_k])
        g_b = self.global_view(b_ptr, dtype=self.operand_dtype, shape=[self.mma_n, self.mma_k])
        g_d = self.global_view(d_ptr, dtype=self.output_dtype, shape=[self.mma_m, self.mma_n])

        s_a = self.shared_tensor(dtype=self.operand_dtype, shape=[self.mma_m, self.mma_k])
        s_b = self.shared_tensor(dtype=self.operand_dtype, shape=[self.mma_n, self.mma_k])
        t_d_storage = self.tcgen05.alloc(
            dtype=self.accumulator_dtype,
            shape=[self.mma_m, cdiv(self.mma_n, self.column_granularity) * self.column_granularity],
            init=0.0,
        )
        t_d = self.tcgen05.slice(t_d_storage, offsets=[0, 0], shape=[self.mma_m, self.mma_n])

        mbarriers = self.mbarrier.alloc(count=[1, 1])
        tma_mbarrier = mbarriers[0]
        mma_mbarrier = mbarriers[1]
        self.sync()

        # load a and b from global to shared
        with self.single_thread():
            self.tma.global_to_shared(
                src=g_a,
                dst=s_a,
                offsets=[0, 0],
                mbarrier=tma_mbarrier,
            )
            self.tma.global_to_shared(
                src=g_b,
                dst=s_b,
                offsets=[0, 0],
                mbarrier=tma_mbarrier,
            )
            self.mbarrier.arrive(tma_mbarrier)
        self.mbarrier.wait(tma_mbarrier, phase=0)

        # perform mma
        self.tcgen05.mma(a=s_a, b=s_b.transpose(), d=t_d)
        self.tcgen05.commit(mma_mbarrier)
        self.mbarrier.wait(mma_mbarrier, phase=0)

        # store d from t_d to global
        r_d_accumulator = self.tcgen05.load(t_d, offsets=[0, 0], shape=[self.mma_m, self.mma_n])
        r_d_output = self.cast(r_d_accumulator, dtype=self.output_dtype)
        self.store_global(g_d, r_d_output, offsets=[0, 0])

        # free tensor memory
        self.tcgen05.dealloc(t_d_storage)


@tilus.testing.requires.nvgpu_sm100a
@pytest.mark.parametrize(
    "operand_dtype, accumulator_dtype, output_dtype, mma_m, mma_n, mma_k",
    [
        (float16, float32, float16, 128, 8, 16),
        (float16, float32, float16, 128, 16, 16),
        (float16, float32, float16, 128, 24, 16),
        (float16, float32, float16, 128, 32, 16),
        (float8_e4m3, float32, float16, 128, 8, 32),
        (float8_e4m3, float32, float16, 128, 16, 32),
        (float8_e4m3, float32, float16, 128, 24, 32),
        (float8_e4m3, float32, float16, 128, 32, 32),
    ],
)
def test_tcgen05_mma(operand_dtype, accumulator_dtype, output_dtype, mma_m, mma_n, mma_k):
    a = torch.randn(mma_m, mma_k, dtype=torch.float16, device="cuda").to(dtype_to_torch(operand_dtype))
    b = torch.randn(mma_n, mma_k, dtype=torch.float16, device="cuda").to(dtype_to_torch(operand_dtype))
    d = torch.empty(mma_m, mma_n, dtype=dtype_to_torch(output_dtype), device="cuda")

    kernel = Tcgen05MmaExample(operand_dtype, accumulator_dtype, output_dtype, mma_m, mma_n, mma_k)
    kernel(a, b, d)
    torch.cuda.synchronize()

    expected = torch.matmul(a.to(torch.float32), b.T.to(torch.float32)).to(dtype_to_torch(output_dtype))
    actual = d
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
