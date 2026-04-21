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
"""End-to-end tests for ``self.scan`` / ``self.cumsum`` / ``self.cumprod`` across layouts.

Covers the layout classes called out in the design matrix:

- pure intra-warp
- pure intra-thread (2D tile, dim in local-only axis)
- intra-thread + intra-warp
- pure inter-warp
- all three phases (intra-thread + intra-warp + inter-warp)
- 2D tiles (scan one axis only)

Plus the op family (add / mul / max / min / and / or / xor), inclusive +
exclusive modes, and the input-is-output (``out=x``) edge case.
"""

import functools

import pytest
import tilus
import torch
from tilus import int32

# ---------------------------------------------------------------------------
# Reference scan implementations (CPU/torch)
# ---------------------------------------------------------------------------


def _identity(op, dtype):
    if op == "add":
        return 0
    if op == "mul":
        return 1
    if op == "max":
        return torch.iinfo(dtype).min if dtype.is_signed else 0
    if op == "min":
        return torch.iinfo(dtype).max
    if op == "and":
        return -1 if dtype.is_signed else torch.iinfo(dtype).max
    if op == "or":
        return 0
    if op == "xor":
        return 0
    raise ValueError(op)


def _combine(op, a, b):
    if op == "add":
        return a + b
    if op == "mul":
        return a * b
    if op == "max":
        return torch.maximum(a, b)
    if op == "min":
        return torch.minimum(a, b)
    if op == "and":
        return torch.bitwise_and(a, b)
    if op == "or":
        return torch.bitwise_or(a, b)
    if op == "xor":
        return torch.bitwise_xor(a, b)
    raise ValueError(op)


def _reference_scan(x: torch.Tensor, dim: int, op: str, exclusive: bool) -> torch.Tensor:
    # Move dim to the front, do the scan, then move back. Keeps the logic 1D.
    x = x.clone()
    x = x.movedim(dim, 0)
    out = torch.empty_like(x)
    acc = torch.full_like(x[0], _identity(op, x.dtype))
    for i in range(x.shape[0]):
        if exclusive:
            out[i] = acc
            acc = _combine(op, acc, x[i])
        else:
            acc = _combine(op, acc, x[i])
            out[i] = acc
    return out.movedim(0, dim).contiguous()


# ---------------------------------------------------------------------------
# Kernels: one class per layout class; op/exclusive/axis are ``__init__`` args
# so the compiler sees them as compile-time constants.
# ---------------------------------------------------------------------------


class _ScanKernel1D(tilus.Script):
    """Generic 1D scan template; the layout is whatever ``load_global`` picks."""

    def __init__(self, n: int, op: str, exclusive: bool, num_warps: int = 1, in_place: bool = False):
        super().__init__()
        self.n = n
        self.op = op
        self.exclusive = exclusive
        self.num_warps = num_warps
        self.in_place = in_place

    def __call__(self, out_ptr: ~int32) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = self.num_warps
        g = self.global_view(out_ptr, dtype=int32, shape=[self.n])
        x = self.load_global(g, offsets=[0], shape=[self.n])
        if self.in_place:
            y = self.scan(x, dim=0, op=self.op, exclusive=self.exclusive, out=x)
        else:
            y = self.scan(x, dim=0, op=self.op, exclusive=self.exclusive)
        self.store_global(g, y, offsets=[0])


class _ScanKernel2D(tilus.Script):
    """2D scan template: scan along ``dim`` of a ``[H, W]`` tile."""

    def __init__(self, h: int, w: int, dim: int, op: str, exclusive: bool, num_warps: int = 1):
        super().__init__()
        self.h = h
        self.w = w
        self.dim = dim
        self.op = op
        self.exclusive = exclusive
        self.num_warps = num_warps

    def __call__(self, out_ptr: ~int32) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = self.num_warps
        g = self.global_view(out_ptr, dtype=int32, shape=[self.h, self.w])
        x = self.load_global(g, offsets=[0, 0], shape=[self.h, self.w])
        y = self.scan(x, dim=self.dim, op=self.op, exclusive=self.exclusive)
        self.store_global(g, y, offsets=[0, 0])


# ---------------------------------------------------------------------------
# Input generators — cover corners that might expose overflow or identity bugs.
# ---------------------------------------------------------------------------


def _input_for_op(op: str, shape, device="cuda") -> torch.Tensor:
    # Use uint32 for bitwise ops to exercise the bitwise path; int32 otherwise.
    dtype = torch.int32
    if op == "mul":
        # Keep values small to avoid int32 overflow in cumprod over 32+ elements.
        return torch.randint(-2, 3, shape, dtype=dtype, device=device)
    if op in ("add",):
        return torch.arange(
            1, int(functools.reduce(lambda a, b: a * b, shape)) + 1, dtype=dtype, device=device
        ).reshape(shape)
    if op in ("max", "min"):
        return torch.randint(-100, 100, shape, dtype=dtype, device=device)
    if op in ("and", "or", "xor"):
        return torch.randint(0, 0xFFFF, shape, dtype=dtype, device=device)
    raise ValueError(op)


# ---------------------------------------------------------------------------
# Parametrized matrix: each row is (label, num_warps, shape, builder)
# ---------------------------------------------------------------------------

_LAYOUT_CASES_1D = [
    pytest.param(32, 1, id="intra_warp_N32"),
    pytest.param(128, 1, id="intra_thread_plus_intra_warp_N128"),
    pytest.param(64, 2, id="inter_warp_N64"),
    pytest.param(256, 2, id="all_three_phases_N256"),
]

_OPS = ["add", "mul", "max", "min", "and", "or", "xor"]


@pytest.mark.parametrize("n, num_warps", _LAYOUT_CASES_1D)
@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize("exclusive", [False, True])
def test_scan_1d(n: int, num_warps: int, op: str, exclusive: bool):
    x = _input_for_op(op, (n,))
    expected = _reference_scan(x, dim=0, op=op, exclusive=exclusive)
    _ScanKernel1D(n=n, op=op, exclusive=exclusive, num_warps=num_warps)(x)
    torch.testing.assert_close(x, expected)


@pytest.mark.parametrize("n, num_warps", _LAYOUT_CASES_1D)
@pytest.mark.parametrize("op", ["add", "max"])  # just a sampler — in-place is orthogonal to op logic
@pytest.mark.parametrize("exclusive", [False, True])
def test_scan_1d_in_place(n: int, num_warps: int, op: str, exclusive: bool):
    """Explicitly pass ``out=x`` so input and output alias the same register tile.

    Exercises the scratch-copy path for inclusive scan where the original
    values need to be preserved across the in-place Blelloch pass.
    """
    x = _input_for_op(op, (n,))
    expected = _reference_scan(x, dim=0, op=op, exclusive=exclusive)
    _ScanKernel1D(n=n, op=op, exclusive=exclusive, num_warps=num_warps, in_place=True)(x)
    torch.testing.assert_close(x, expected)


# 2D: scan along either axis. Pick a tile that splits nicely across threads.
_LAYOUT_CASES_2D = [
    pytest.param(4, 32, 1, id="H4W32_single_warp_scan_W"),
    pytest.param(4, 32, 1, id="H4W32_single_warp_scan_W_repeat"),  # same — kept for explicit dim param below
    pytest.param(8, 32, 2, id="H8W32_two_warps"),
]


@pytest.mark.parametrize("h, w, num_warps", [(4, 32, 1), (8, 32, 2)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("op", ["add", "max", "xor"])
@pytest.mark.parametrize("exclusive", [False, True])
def test_scan_2d(h: int, w: int, num_warps: int, dim: int, op: str, exclusive: bool):
    x = _input_for_op(op, (h, w))
    expected = _reference_scan(x, dim=dim, op=op, exclusive=exclusive)
    _ScanKernel2D(h=h, w=w, dim=dim, op=op, exclusive=exclusive, num_warps=num_warps)(x)
    torch.testing.assert_close(x, expected)


# ---------------------------------------------------------------------------
# Shortcut methods
# ---------------------------------------------------------------------------


class _CumsumKernel(tilus.Script):
    def __init__(self, n: int, exclusive: bool = False):
        super().__init__()
        self.n = n
        self.exclusive = exclusive

    def __call__(self, out_ptr: ~int32) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = 1
        g = self.global_view(out_ptr, dtype=int32, shape=[self.n])
        x = self.load_global(g, offsets=[0], shape=[self.n])
        y = self.cumsum(x, dim=0, exclusive=self.exclusive)
        self.store_global(g, y, offsets=[0])


class _CumprodKernel(tilus.Script):
    def __init__(self, n: int, exclusive: bool = False):
        super().__init__()
        self.n = n
        self.exclusive = exclusive

    def __call__(self, out_ptr: ~int32) -> None:
        self.attrs.blocks = 1
        self.attrs.warps = 1
        g = self.global_view(out_ptr, dtype=int32, shape=[self.n])
        x = self.load_global(g, offsets=[0], shape=[self.n])
        y = self.cumprod(x, dim=0, exclusive=self.exclusive)
        self.store_global(g, y, offsets=[0])


def test_cumsum_shortcut():
    x = torch.arange(1, 33, dtype=torch.int32, device="cuda")
    expected = _reference_scan(x, dim=0, op="add", exclusive=False)
    _CumsumKernel(n=32)(x)
    torch.testing.assert_close(x, expected)


def test_cumprod_shortcut():
    x = torch.randint(-2, 3, (32,), dtype=torch.int32, device="cuda")
    expected = _reference_scan(x, dim=0, op="mul", exclusive=False)
    _CumprodKernel(n=32)(x)
    torch.testing.assert_close(x, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
