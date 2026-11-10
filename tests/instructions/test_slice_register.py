# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import re
from pathlib import Path

import pytest
import tilus
import torch
from tilus import int32
from tilus.backends.codegen import generate_ir_module
from tilus.drivers import BuildOptions, optimize_ir_module, optimize_program
from tilus.hidet.backend.codegen import codegen
from tilus.ir.layout.ops import local, replicated, spatial
from tilus.target import nvgpu_sm80, scope
from tilus.testing import requires


class ReadSlice(tilus.Script):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x: ~int32, y: ~int32):
        self.attrs.blocks = 1
        self.attrs.warps = 1

        g_x = self.global_view(x, dtype=int32, shape=[4, 4])
        g_y = self.global_view(y, dtype=int32, shape=[4])
        with self.thread_group(0, num_threads=1):
            r = self.load_global(g_x, offsets=[0, 0], shape=[4, 4])
            self.annotate_layout(r, local(4, 4))
            if self.dim == 0:
                s = r[1, :]
            else:
                s = r[:, 1]
            self.store_global(g_y, s, offsets=[0])


class AssignSlice(tilus.Script):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x: ~int32, y: ~int32):
        self.attrs.blocks = 1
        self.attrs.warps = 1

        g_x = self.global_view(x, dtype=int32, shape=[4, 4])
        g_y = self.global_view(y, dtype=int32, shape=[4])
        with self.thread_group(0, num_threads=1):
            r = self.load_global(g_x, offsets=[0, 0], shape=[4, 4])
            self.annotate_layout(r, local(4, 4))
            v = self.load_global(g_y, offsets=[0], shape=[4])
            self.annotate_layout(v, local(4))
            if self.dim == 0:
                r[1, :] = v
            else:
                r[:, 1] = v
            self.store_global(g_x, r, offsets=[0, 0])


class ReadColumnAcrossThreads(tilus.Script):
    def __call__(self, x: ~int32, y: ~int32):
        self.attrs.blocks = 1
        self.attrs.warps = 1

        g_x = self.global_view(x, dtype=int32, shape=[4, 4])
        g_y = self.global_view(y, dtype=int32, shape=[4])
        with self.thread_group(0, num_threads=4):
            r = self.load_global(g_x, offsets=[0, 0], shape=[4, 4])
            self.annotate_layout(r, spatial(4, 1).local(1, 4))
            self.store_global(g_y, r[:, 1], offsets=[0])


class AssignAcrossThreads(tilus.Script):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def __call__(self, x: ~int32, y: ~int32):
        self.attrs.blocks = 1
        self.attrs.warps = 1

        g_x = self.global_view(x, dtype=int32, shape=[4, 4])
        g_y = self.global_view(y, dtype=int32, shape=[4])
        with self.thread_group(0, num_threads=4):
            r = self.load_global(g_x, offsets=[0, 0], shape=[4, 4])
            self.annotate_layout(r, spatial(4, 1).local(1, 4))
            v = self.load_global(g_y, offsets=[0], shape=[4])
            if self.dim == 0:
                # the dropped row dim is distributed across threads: only the owner of row 1 may write
                self.annotate_layout(v, replicated(4, num_workers=4))
                r[1, :] = v
            else:
                self.annotate_layout(v, spatial(4))
                r[:, 1] = v
            self.store_global(g_x, r, offsets=[0, 0])


class Slice3D(tilus.Script):
    """Read and assign r[:, 1, :], whose retained dims (0 and 2) are not contiguous."""

    def __call__(self, x: ~int32, y: ~int32, z: ~int32):
        self.attrs.blocks = 1
        self.attrs.warps = 1

        g_x = self.global_view(x, dtype=int32, shape=[2, 3, 4])
        g_y = self.global_view(y, dtype=int32, shape=[2, 4])
        g_z = self.global_view(z, dtype=int32, shape=[2, 4])
        with self.thread_group(0, num_threads=1):
            r = self.load_global(g_x, offsets=[0, 0, 0], shape=[2, 3, 4])
            self.annotate_layout(r, local(2, 3, 4))
            self.store_global(g_z, r[:, 1, :], offsets=[0, 0])
            v = self.load_global(g_y, offsets=[0, 0], shape=[2, 4])
            self.annotate_layout(v, local(2, 4))
            r[:, 1, :] = v
            self.store_global(g_x, r, offsets=[0, 0, 0])


def generated_source(script, tmp_path: Path) -> str:
    with scope(nvgpu_sm80):
        program = script._jit_instance_for(None, None).transpiled_programs[0]  # type: ignore
        ir_module = generate_ir_module(optimize_program(program, BuildOptions(), tmp_path))
        ir_module = optimize_ir_module(ir_module, tmp_path)
        codegen(ir_module, src_out_path=str(tmp_path / "source.cu"), target="cuda")
    return (tmp_path / "source.cu").read_text()


def copied_indices(source: str, pattern: str) -> list[int]:
    """Evaluate the register index of the matching element copy for each iteration of its loop."""
    match = re.search(pattern, source)
    assert match is not None, f"no element copy matching {pattern!r} in the generated source"
    return [eval(match["expr"], {match["var"]: i}) for i in range(4)]


@pytest.mark.parametrize("dim, expected", [(0, [4, 5, 6, 7]), (1, [1, 5, 9, 13])])
def test_read_slice_codegen(dim, expected, tmp_path: Path):
    source = generated_source(ReadSlice(dim), tmp_path)
    assert copied_indices(source, r"slice_regs\[(?P<var>\w+)\] = regs\[(?P<expr>.+)\];") == expected


@pytest.mark.parametrize("dim, expected", [(0, [4, 5, 6, 7]), (1, [1, 5, 9, 13])])
def test_assign_slice_codegen(dim, expected, tmp_path: Path):
    source = generated_source(AssignSlice(dim), tmp_path)
    assert copied_indices(source, r"regs\[(?P<expr>.+)\] = regs_1\[(?P<var>\w+)\];") == expected


def matrix() -> torch.Tensor:
    return torch.arange(16, dtype=torch.int32, device="cuda").reshape(4, 4)


@requires.nvgpu_sm80
@pytest.mark.parametrize("dim", [0, 1])
def test_read_slice(dim):
    x = matrix()
    y = torch.full((4,), -1, dtype=torch.int32, device="cuda")

    ReadSlice(dim)(x, y)

    torch.testing.assert_close(y, x[1, :] if dim == 0 else x[:, 1])


@requires.nvgpu_sm80
def test_read_column_across_threads():
    x = matrix()
    y = torch.full((4,), -1, dtype=torch.int32, device="cuda")

    ReadColumnAcrossThreads()(x, y)

    torch.testing.assert_close(y, x[:, 1])


@requires.nvgpu_sm80
@pytest.mark.parametrize("dim", [0, 1])
def test_assign_slice(dim):
    x = matrix()
    y = torch.arange(100, 104, dtype=torch.int32, device="cuda")
    expected = x.clone()
    if dim == 0:
        expected[1, :] = y
    else:
        expected[:, 1] = y

    AssignSlice(dim)(x, y)

    torch.testing.assert_close(x, expected)


@requires.nvgpu_sm80
@pytest.mark.parametrize("dim", [0, 1])
def test_assign_slice_across_threads(dim):
    x = matrix()
    y = torch.arange(100, 104, dtype=torch.int32, device="cuda")
    expected = x.clone()
    if dim == 0:
        expected[1, :] = y
    else:
        expected[:, 1] = y

    AssignAcrossThreads(dim)(x, y)

    torch.testing.assert_close(x, expected)


@requires.nvgpu_sm80
def test_slice_3d_non_contiguous_dims():
    x = torch.arange(24, dtype=torch.int32, device="cuda").reshape(2, 3, 4)
    y = torch.arange(100, 108, dtype=torch.int32, device="cuda").reshape(2, 4)
    z = torch.full((2, 4), -1, dtype=torch.int32, device="cuda")
    expected_x = x.clone()
    expected_x[:, 1, :] = y
    expected_z = x[:, 1, :].clone()

    Slice3D()(x, y, z)

    torch.testing.assert_close(z, expected_z)
    torch.testing.assert_close(x, expected_x)
