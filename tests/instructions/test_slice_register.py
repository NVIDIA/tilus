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
from tilus.ir.layout.ops import local, spatial
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
    return [eval(match["expr"], {match["var"]: i}) for i in range(4)]


@pytest.mark.parametrize("dim, expected", [(0, [4, 5, 6, 7]), (1, [1, 5, 9, 13])])
def test_read_slice_codegen(dim, expected, tmp_path: Path):
    source = generated_source(ReadSlice(dim), tmp_path)
    assert copied_indices(source, r"slice_regs\[(?P<var>\w+)\] = regs\[(?P<expr>.+)\];") == expected


@pytest.mark.parametrize("dim, expected", [(0, [4, 5, 6, 7]), (1, [1, 5, 9, 13])])
def test_assign_slice_codegen(dim, expected, tmp_path: Path):
    source = generated_source(AssignSlice(dim), tmp_path)
    assert copied_indices(source, r"regs\[(?P<expr>.+)\] = regs_1\[(?P<var>\w+)\];") == expected


def matrix(device: str) -> torch.Tensor:
    return torch.arange(16, dtype=torch.int32, device=device).reshape(4, 4)


@requires.nvgpu_sm80
@pytest.mark.parametrize("dim", [0, 1])
def test_read_slice(dim):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = matrix(device)
    y = torch.full((4,), -1, dtype=torch.int32, device=device)

    ReadSlice(dim)(x, y)

    torch.testing.assert_close(y, x[1, :] if dim == 0 else x[:, 1])


@requires.nvgpu_sm80
def test_read_column_across_threads():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = matrix(device)
    y = torch.full((4,), -1, dtype=torch.int32, device=device)

    ReadColumnAcrossThreads()(x, y)

    torch.testing.assert_close(y, x[:, 1])


@requires.nvgpu_sm80
@pytest.mark.parametrize("dim", [0, 1])
def test_assign_slice(dim):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = matrix(device)
    y = torch.arange(100, 104, dtype=torch.int32, device=device)
    expected = x.clone()
    if dim == 0:
        expected[1, :] = y
    else:
        expected[:, 1] = y

    AssignSlice(dim)(x, y)

    torch.testing.assert_close(x, expected)
