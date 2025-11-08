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
from tilus import boolean, int32
from tilus.ir.layout import RegisterLayout
from tilus.ir.layout.ops import local, spatial
from tilus.utils import cdiv


class TestWhereKernel(tilus.Script):
    def __init__(self, layout: RegisterLayout):
        super().__init__()
        self.layout = layout
        self.block_m = layout.shape[0]
        self.block_n = layout.shape[1]
        self.num_warps = layout.spatial_size // 32

    def __call__(self, m: int32, n: int32, cond_ptr: ~boolean, x_ptr: ~int32, y_ptr: ~int32, out_ptr: ~int32) -> None:
        self.attrs.warps = self.num_warps
        self.attrs.blocks = cdiv(m, self.block_m), cdiv(n, self.block_n)

        m_offset: int32 = self.blockIdx.x * self.block_m
        n_offset: int32 = self.blockIdx.y * self.block_n

        gc = self.global_view(cond_ptr, dtype=boolean, shape=(m, n))
        gx = self.global_view(x_ptr, dtype=int32, shape=(m, n))
        gy = self.global_view(y_ptr, dtype=int32, shape=(m, n))
        go = self.global_view(out_ptr, dtype=int32, shape=(m, n))

        rc = self.load_global(gc, offsets=[m_offset, n_offset], shape=self.layout.shape)
        rx = self.load_global(gx, offsets=[m_offset, n_offset], shape=[self.block_m, self.block_n])
        ry = self.load_global(gy, offsets=[m_offset, n_offset], shape=[self.block_m, self.block_n])
        ro = self.where(rc, rx, ry)

        self.store_global(go, ro, offsets=[m_offset, n_offset])

        self.annotate_layout(rc, self.layout)


@pytest.mark.parametrize("m, n, layout", [[16, 16, spatial(4, 8)], [128, 128, local(2, 2).spatial(4, 8).local(2, 2)]])
def test_where(
    m: int,
    n: int,
    layout: RegisterLayout,
) -> None:
    cond = torch.randint(0, 2, (m, n), dtype=torch.bool).cuda()
    x = torch.arange(m * n, dtype=torch.int32).reshape((m, n)).cuda()
    y = torch.arange(m * n, dtype=torch.int32).reshape((m, n)).cuda() + m * n
    expected = torch.where(cond, x, y)

    actual = torch.empty_like(x)
    kernel = TestWhereKernel(layout)
    kernel(m, n, cond, x, y, actual)

    assert torch.allclose(actual, expected), f"Failed for layout {layout} with m={m}, n={n}"


if __name__ == "__main__":
    pytest.main([__file__])
