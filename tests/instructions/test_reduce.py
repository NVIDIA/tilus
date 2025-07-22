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
from tilus import int32
from tilus.ir.layout import RegisterLayout, register_layout, spatial


class TestReduceKernel(tilus.Script):
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
            init=lambda indices: indices[0] * self.layout.shape[1] + indices[1],
        )
        b = self.sum(a, dim=self.dim, keepdim=True)
        g_out = self.global_view(ptr=out_ptr, dtype=int32, shape=b.shape)
        self.store_global(g_out, b, offsets=[0, 0], dims=[0, 1])


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
    demo = TestReduceKernel(layout, dim=dim)
    demo(actual)
    assert torch.allclose(actual, expected), f"Failed for layout {layout} and dim {dim}"
