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
from tilus import float8_e4m3, float16, float32, void_p
from tilus.hidet.ir.type import DataType
from tilus.utils import dtype_to_torch


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

    def __call__(self, a_ptr: void_p, b_ptr: void_p, d_ptr: void_p) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = 4

        g_a = self.global_view(a_ptr, dtype=self.operand_dtype, shape=[self.mma_m, self.mma_k])
        g_b = self.global_view(b_ptr, dtype=self.operand_dtype, shape=[self.mma_n, self.mma_k])
        g_d = self.global_view(d_ptr, dtype=self.output_dtype, shape=[self.mma_m, self.mma_n])

        s_a = self.shared_tensor(dtype=self.operand_dtype, shape=[self.mma_m, self.mma_k])
        s_b = self.shared_tensor(dtype=self.operand_dtype, shape=[self.mma_n, self.mma_k])
        t_d = self.tcgen05.alloc(dtype=self.accumulator_dtype, shape=[self.mma_m, self.mma_n])

        mbarriers = self.mbarrier.alloc(counts=[1, 1])
        tma_mbarrier = mbarriers[0]
        mma_mbarrier = mbarriers[1]
        self.sync()

        # load a and b from global to shared
        with self.single_thread():
            self.mbarrier.arrive_and_expect_tx(tma_mbarrier, transaction_bytes=s_a.nbytes + s_b.nbytes)
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
        self.mbarrier.wait(tma_mbarrier, phase=0)

        # perform mma
        with self.single_thread():
            self.tcgen05.mma(a=s_a, b=s_b.transpose(), d=t_d, enable_input_d=False)
            self.tcgen05.commit(mma_mbarrier)
        self.mbarrier.wait(mma_mbarrier, phase=0)

        # store d from t_d to global
        r_d_accumulator = self.tcgen05.load(t_d)
        r_d_output = self.cast(r_d_accumulator, dtype=self.output_dtype)
        self.store_global(g_d, r_d_output, offsets=[0, 0])

        # free tensor memory
        self.tcgen05.dealloc(t_d)


class Tcgen05Mma2CTAExample(tilus.Script):
    """
    The example demonstrates the use of the tcgen05 MMA instruction with 2 CTAs working collaboratively on a single MMA operation.

    cta_group=2:  D = A @ B + D

    Each CTA provides its own slice of A, B, and D.
    CTA0 = CTA with last bit of cluster rank = 0
    CTA1 = CTA with last bit of cluster rank = 1

    Input A (M, K)      Input B (K, N)          Output D (M, N)
    ┌──────────────┐    ┌───────┬───────┐      ┌───────────────┐
    │              │    │       │       │      │               │
    │  a0 (M/2, K) │    │  b0   │  b1   │      │  d0 (M/2, N)  │
    │  [CTA0]      │    │(K,N/2)│(K,N/2)│      │  [CTA0]       │
    │              │    │[CTA0] │[CTA1] │      │               │
    ├──────────────┤    │       │       │      ├───────────────┤
    │              │    │       │       │      │               │
    │  a1 (M/2, K) │    │       │       │      │  d1 (M/2, N)  │
    │  [CTA1]      │    │       │       │      │  [CTA1]       │
    │              │    │       │       │      │               │
    └──────────────┘    └───────┴───────┘      └───────────────┘
    """

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

    def __call__(self, a_ptr: void_p, b_ptr: void_p, d_ptr: void_p) -> None:
        self.attrs.blocks = (2, 1, 1)
        self.attrs.cluster_blocks = (2, 1, 1)
        self.attrs.warps = 4

        g_a = self.global_view(a_ptr, dtype=self.operand_dtype, shape=[self.mma_m, self.mma_k])
        g_b = self.global_view(b_ptr, dtype=self.operand_dtype, shape=[self.mma_n, self.mma_k])
        g_d = self.global_view(d_ptr, dtype=self.output_dtype, shape=[self.mma_m, self.mma_n])

        s_a = self.shared_tensor(dtype=self.operand_dtype, shape=[self.mma_m // 2, self.mma_k])
        s_b = self.shared_tensor(dtype=self.operand_dtype, shape=[self.mma_n // 2, self.mma_k])
        t_d = self.tcgen05.alloc(dtype=self.accumulator_dtype, shape=[self.mma_m // 2, self.mma_n], cta_group=2)

        mbarriers = self.mbarrier.alloc(counts=[1, 1])
        tma_mbarrier = mbarriers[0]
        mma_mbarrier = mbarriers[1]
        self.cluster.sync()

        cta_rank = self.cluster.blockRank

        if cta_rank == 0:
            with self.single_thread():
                self.mbarrier.arrive_and_expect_tx(tma_mbarrier, transaction_bytes=(s_a.nbytes + s_b.nbytes) * 2)
        else:
            tma_mbarrier = self.cluster.map_shared_addr(tma_mbarrier, target_rank=cta_rank - 1)

        # load a and b from global to shared
        m_offset = self.mma_m // 2 * cta_rank
        n_offset = self.mma_n // 2 * cta_rank
        with self.single_thread():
            self.tma.global_to_shared(
                src=g_a,
                dst=s_a,
                offsets=[m_offset, 0],
                mbarrier=tma_mbarrier,
                cta_group=2,
            )
            self.tma.global_to_shared(
                src=g_b,
                dst=s_b,
                offsets=[n_offset, 0],
                mbarrier=tma_mbarrier,
                cta_group=2,
            )

        # perform mma
        if cta_rank == 0:
            self.mbarrier.wait(tma_mbarrier, phase=0)
            with self.single_thread():
                self.tcgen05.mma(a=s_a, b=s_b.transpose(), d=t_d, cta_group=2, enable_input_d=False)
                self.tcgen05.commit(mma_mbarrier, cta_group=2, multicast_mask=0b11)
        self.mbarrier.wait(mma_mbarrier, phase=0)

        # store d from t_d to global
        r_d_accumulator = self.tcgen05.load(t_d)
        r_d_output = self.cast(r_d_accumulator, dtype=self.output_dtype)
        self.store_global(g_d, r_d_output, offsets=[m_offset, 0])

        # free tensor memory
        self.tcgen05.dealloc(t_d)


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


@tilus.testing.requires.nvgpu_sm100a
@pytest.mark.parametrize(
    "operand_dtype, accumulator_dtype, output_dtype, mma_m, mma_n, mma_k",
    [
        (float16, float32, float16, 256, 16, 16),
        (float16, float32, float16, 256, 32, 16),
        (float16, float32, float16, 256, 48, 16),
        (float16, float32, float16, 256, 64, 16),
        (float8_e4m3, float32, float16, 256, 16, 32),
        (float8_e4m3, float32, float16, 256, 32, 32),
        (float8_e4m3, float32, float16, 256, 48, 32),
        (float8_e4m3, float32, float16, 256, 64, 32),
    ],
)
def test_tcgen05_mma_2cta(operand_dtype, accumulator_dtype, output_dtype, mma_m, mma_n, mma_k):
    a = torch.randn(mma_m, mma_k, dtype=torch.float16, device="cuda").to(dtype_to_torch(operand_dtype))
    b = torch.randn(mma_n, mma_k, dtype=torch.float16, device="cuda").to(dtype_to_torch(operand_dtype))
    d = torch.empty(mma_m, mma_n, dtype=dtype_to_torch(output_dtype), device="cuda")

    kernel = Tcgen05Mma2CTAExample(operand_dtype, accumulator_dtype, output_dtype, mma_m, mma_n, mma_k)
    kernel(a, b, d)
    torch.cuda.synchronize()

    expected = torch.matmul(a.to(torch.float32), b.T.to(torch.float32)).to(dtype_to_torch(output_dtype))
    actual = d
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
