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
from tilus.ir.layout import RegisterLayout, register_layout
from tilus.ir.layout.ops import spatial


class ReduceKernelExample(tilus.Script):
    def __init__(self, layout: RegisterLayout, dim=0):
        super().__init__()
        self.layout = layout
        self.dim = dim

    def __call__(self, out_ptr: ~int32):
        self.attrs.blocks = 1
        self.attrs.warps = self.layout.spatial_size // 32

        a = self.register_tensor(
            dtype=int32,
            shape=self.layout.shape,
            layout=self.layout,
            init=lambda i, j: i * self.layout.shape[1] + j,
        )
        b = self.sum(a, dim=self.dim, keepdim=True)
        g_out = self.global_view(ptr=out_ptr, dtype=int32, shape=b.shape)
        self.store_global(g_out, b, offsets=[0, 0], dims=[0, 1])


class AnyAllInstExample(tilus.Script):
    def __call__(self, x_ptr: ~int32, y_ptr: ~boolean):
        self.attrs.blocks = 1
        self.attrs.warps = 1

        g_x = self.global_view(ptr=x_ptr, dtype=int32, shape=(32, 32))
        g_y = self.global_view(ptr=y_ptr, dtype=boolean, shape=[2])
        r_x = self.load_global(g_x, offsets=[0, 0], shape=[32, 32])

        self.store_global(g_y, src=self.any(r_x != 0), offsets=[0], dims=[])
        self.store_global(g_y, src=self.all(r_x != 0), offsets=[1], dims=[])


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize(
    "layout",
    [
        spatial(4, 8),
        spatial(4, 8).local(1, 2),
        spatial(2, 2).local(1, 2).spatial(2, 1).local(2, 1).spatial(2, 2),
        spatial(4, 32),
        spatial(2, 4).spatial(2, 4),
        spatial(2, 4).column_spatial(2, 4),
        spatial(2, 4).spatial(2, 4).column_spatial(2, 1),
        spatial(4, 4).spatial(2, 4).column_spatial(2, 1),
        spatial(2, 4).spatial(4, 8),
        spatial(2, 4).spatial(4, 8).local(2, 2),
        spatial(2, 4).local(2, 2).spatial(4, 8).local(2, 2),
        register_layout(
            shape=[32, 16], mode_shape=[2, 2, 8, 2, 4, 2], spatial_modes=[-4, 2, 4], local_modes=[0, 3, 1, 5]
        ),
    ],
)
def test_reduce_instruction(dim: int, layout: RegisterLayout):
    shape = layout.shape
    original_tensor = torch.arange(shape[0] * shape[1]).cuda().reshape(shape)
    expected = original_tensor.sum(dim=dim).to(torch.int32)
    actual = torch.empty_like(expected)
    demo = ReduceKernelExample(layout, dim=dim)
    demo(actual)
    assert torch.allclose(actual, expected), f"Failed for layout {layout} and dim {dim}"


def test_any_all_reduce_instruction():
    kernel = AnyAllInstExample()
    x0 = torch.zeros((32, 32), dtype=torch.int32).cuda()
    y0 = torch.asarray([False, False], dtype=torch.bool).cuda()
    x1 = torch.ones((32, 32), dtype=torch.int32).cuda()
    y1 = torch.asarray([True, True], dtype=torch.bool).cuda()
    x2 = torch.randint(0, 2, size=(32, 32), dtype=torch.int32).cuda()
    x2[0, 0] = 1
    x2[0, 1] = 0
    y2 = torch.asarray([True, False], dtype=torch.bool).cuda()
    for x, y in zip([x0, x1, x2], [y0, y1, y2]):
        y_actual = torch.empty_like(y)
        kernel(x, y_actual)
        assert torch.allclose(y_actual, y), f"Failed for x={x} and y={y}, y_actual={y_actual}"
