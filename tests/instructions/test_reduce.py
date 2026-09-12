# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tilus import bfloat16, boolean, float16, float32, int32
from tilus.ir.layout import RegisterLayout, register_layout
from tilus.ir.layout.ops import replicated, spatial


class ReduceKernelExample(tilus.Script):
    def __init__(self, layout: RegisterLayout, dim=0):
        super().__init__()
        self.layout = layout
        self.dim = dim

    def __call__(self, out_ptr: ~int32) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = self.layout.spatial_size // 32

        a = self.register_tensor(
            dtype=int32,
            shape=self.layout.shape,
            init=lambda i, j: i * self.layout.shape[1] + j,
        )
        b = self.sum(a, dim=self.dim, keepdim=True)
        g_out = self.global_view(ptr=out_ptr, dtype=int32, shape=b.shape)
        self.store_global(g_out, b, offsets=[0, 0], dims=[0, 1])

        self.annotate_layout(a, self.layout)


class AnyAllInstExample(tilus.Script):
    def __call__(self, x_ptr: ~int32, y_ptr: ~boolean) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = 1

        g_x = self.global_view(ptr=x_ptr, dtype=int32, shape=(32, 32))
        g_y = self.global_view(ptr=y_ptr, dtype=boolean, shape=[2])
        r_x = self.load_global(g_x, offsets=[0, 0], shape=[32, 32])

        self.store_global(g_y, src=self.any(r_x != 0), offsets=[0], dims=[])
        self.store_global(g_y, src=self.all(r_x != 0), offsets=[1], dims=[])


class IntraWarpReductionMatrixExample(tilus.Script):
    """Expose the reduction result from every lane in each warp-local group."""

    def __init__(self, lane_width: int, num_warps: int, op: str, dtype):
        super().__init__()
        self.lane_width = lane_width
        self.num_warps = num_warps
        self.op = op
        self.dtype = dtype
        self.is_boolean = dtype == boolean
        # The negative spatial mode replicates one logical reduction group across
        # all other lanes.  Thus b[0] is defined in every thread and a store per
        # physical lane verifies the broadcast part of the reduction contract.
        self.layout = replicated(num_workers=32 * num_warps // lane_width) * spatial(lane_width)

    def __call__(self, out_ptr: ~int32) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = self.num_warps

        if self.is_boolean:
            a = self.register_tensor(dtype=self.dtype, shape=[self.lane_width], init=lambda i: (i % 2) == 0)
        else:
            a = self.register_tensor(dtype=self.dtype, shape=[self.lane_width], init=lambda _i: 1)
        if self.op == "sum":
            b = self.sum(a, dim=0, keepdim=True)
        elif self.op == "max":
            b = self.max(a, dim=0, keepdim=True)
        elif self.op == "min":
            b = self.min(a, dim=0, keepdim=True)
        elif self.op == "any":
            b = self.any(a, dim=0, keepdim=True)
        elif self.op == "all":
            b = self.all(a, dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported operation: {self.op}")

        g_out = self.global_view(ptr=out_ptr, dtype=int32, shape=[32 * self.num_warps])
        self.store_global(g_out, b[0].to(int32), offsets=[self.get_thread_binding()], dims=[])
        self.annotate_layout(a, self.layout)


class InterWarpReductionMatrixExample(tilus.Script):
    """Exercise the shared-memory inter-warp path in addition to XOR shuffles."""

    def __init__(self, op: str, dtype):
        super().__init__()
        self.op = op
        self.dtype = dtype
        self.is_boolean = dtype == boolean

    def __call__(self, out_ptr: ~int32) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = 2
        layout = spatial(2, 32)
        if self.is_boolean:
            a = self.register_tensor(dtype=self.dtype, shape=layout.shape, init=lambda i, j: (i + j) % 2 == 0)
        else:
            a = self.register_tensor(dtype=self.dtype, shape=layout.shape, init=lambda _i, _j: 1)
        if self.op == "sum":
            b = self.sum(a, dim=0, keepdim=True)
        elif self.op == "max":
            b = self.max(a, dim=0, keepdim=True)
        elif self.op == "min":
            b = self.min(a, dim=0, keepdim=True)
        elif self.op == "any":
            b = self.any(a, dim=0, keepdim=True)
        elif self.op == "all":
            b = self.all(a, dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported operation: {self.op}")

        g_out = self.global_view(ptr=out_ptr, dtype=int32, shape=b.shape)
        self.store_global(g_out, b.to(int32), offsets=[0, 0], dims=[0, 1])
        self.annotate_layout(a, layout)


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


@pytest.mark.parametrize("lane_width", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("num_warps", [1, 2], ids=["single_warp", "multi_warp"])
@pytest.mark.parametrize(
    ("op", "dtype", "expected"),
    [
        ("sum", int32, lambda width: width),
        ("sum", float32, lambda width: width),
        ("max", float16, lambda _width: 1),
        ("min", bfloat16, lambda _width: 1),
        ("any", boolean, lambda _width: 1),
        ("all", boolean, lambda _width: 0),
    ],
    ids=["sum_int32", "sum_fp32", "max_fp16", "min_bf16", "any", "all"],
)
def test_intra_warp_reduction_equivalence_matrix(lane_width: int, num_warps: int, op: str, dtype, expected):
    """All lanes must agree for every supported XOR-reduction subgroup width."""
    actual = torch.empty(32 * num_warps, dtype=torch.int32, device="cuda")
    IntraWarpReductionMatrixExample(lane_width, num_warps, op, dtype)(actual)
    torch.testing.assert_close(actual, torch.full_like(actual, expected(lane_width)))


@pytest.mark.parametrize(
    ("op", "dtype", "expected"),
    [
        ("sum", int32, 2),
        ("sum", float32, 2),
        ("max", float16, 1),
        ("min", bfloat16, 1),
        ("any", boolean, 1),
        ("all", boolean, 0),
    ],
    ids=["sum_int32", "sum_fp32", "max_fp16", "min_bf16", "any", "all"],
)
def test_inter_warp_reduction_equivalence_matrix(op: str, dtype, expected: int):
    """The shared-memory handoff preserves the same result for two warps."""
    actual = torch.empty((1, 32), dtype=torch.int32, device="cuda")
    InterWarpReductionMatrixExample(op, dtype)(actual)
    torch.testing.assert_close(actual, torch.full_like(actual, expected))
