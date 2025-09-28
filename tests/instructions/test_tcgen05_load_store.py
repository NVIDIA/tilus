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
from tilus import DataType, float16, float32, int32, void_p
from tilus.utils import cdiv


class Tcgen05LoadStoreExample(tilus.Script):
    def __init__(self, dtype: DataType, block_m: int, block_n: int):
        super().__init__()
        self.dtype = dtype
        self.block_m = block_m
        self.block_n = block_n

        assert block_m in (64, 128)

    def __call__(self, m_size: int, n_size: int, x_ptr: void_p, y_ptr: void_p):
        self.attrs.blocks = cdiv(m_size, self.block_m), cdiv(n_size, self.block_n)
        self.attrs.warps = 4

        g_x = self.global_view(x_ptr, dtype=self.dtype, shape=[m_size, n_size])
        g_y = self.global_view(y_ptr, dtype=self.dtype, shape=[m_size, n_size])
        t_a = self.tcgen05.alloc(dtype=self.dtype, shape=[128, self.block_n])

        m_offset: int32 = self.blockIdx.x * self.block_m
        n_offset: int32 = self.blockIdx.y * self.block_n

        # store and load
        r_a = self.load_global(g_x, offsets=[m_offset, n_offset], shape=[self.block_m, self.block_n])

        with self.thread_group(group_index=0, group_size=self.block_m):
            self.tcgen05.store(t_a, src=r_a, offsets=[0, 0])
            self.tcgen05.wait_store()
            r_a_loaded = self.tcgen05.load(t_a, offsets=[0, 0], shape=[self.block_m, self.block_n])
            self.tcgen05.wait_load()
            r_a_loaded += 1
            self.store_global(g_y, src=r_a_loaded, offsets=[m_offset, n_offset])
        self.sync()
        self.tcgen05.dealloc(t_a)


@tilus.testing.requires.nvgpu_sm100
@pytest.mark.parametrize("dtype", [int32, float32, float16])
@pytest.mark.parametrize("block_m", [64, 128])
@pytest.mark.parametrize("shape", [(128, 64), (234, 567), (1234, 2345)])
def test_tcgen_load_store(dtype, block_m, shape):
    kernel = Tcgen05LoadStoreExample(dtype=dtype, block_m=block_m, block_n=64)
    m_size, n_size = shape

    if dtype.is_integer():
        x = torch.randint(0, 100, shape, dtype=getattr(torch, dtype.name), device="cuda")
    elif dtype.is_float():
        x = torch.randn(shape, dtype=getattr(torch, dtype.name), device="cuda")
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    y = torch.zeros_like(x)

    kernel(m_size, n_size, x.data_ptr(), y.data_ptr())
    torch.cuda.synchronize()

    print("x: ", x)
    print("y: ", y)

    torch.testing.assert_close(x + 1, y)


if __name__ == "__main__":
    pytest.main([__file__])
