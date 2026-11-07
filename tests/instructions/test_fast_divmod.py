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
import typing

import pytest
import tilus
import torch
from tilus import int32
from tilus.hidet.ir.dtypes import uint32
from tilus.hidet.lang import attrs, script
from tilus.hidet.lang.script import script_module
from tilus.target import nvgpu_sm80, scope
from tilus.testing import requires

MAX_DIVIDEND = 2**31 - 1
# Every small divisor, plus the divisors around each power of two up to the int32 limit.
DIVISORS = sorted({*range(1, 1025), *(2**k + 1 for k in range(11, 31)), *(2**k - 1 for k in range(11, 32))})


def build_precompute_kernel():
    from tilus.hidet.ir.primitives.cuda.fast_divmod import fastdiv_precompute_m, fastdiv_precompute_s

    with script_module() as module:

        @typing.no_type_check
        @script
        def _precompute(m: ~uint32, s: ~uint32, b: ~uint32, n: int32):
            attrs.func_kind = "cpu_kernel"
            for i in range(n):
                m[i] = fastdiv_precompute_m(b[i])
                s[i] = fastdiv_precompute_s(b[i])

    with scope(nvgpu_sm80):
        return module.build()


def fastdiv_runtime(a: torch.Tensor, multiplier: int, shift: int) -> torch.Tensor:
    """Evaluate cuda_fastdiv_runtime on the host for non-negative dividends."""
    if multiplier == 0:
        return a >> shift
    return ((a * multiplier) >> 32) >> shift


def boundary_dividends(b: int) -> torch.Tensor:
    """Multiples of b and their predecessors at both ends of the dividend range, where the error would show up first."""
    top = MAX_DIVIDEND // b
    factors = torch.cat([torch.arange(1, 17), torch.arange(max(top - 15, 1), top + 1)])
    multiples = factors[factors <= top] * b
    return torch.cat([torch.tensor([0, MAX_DIVIDEND]), multiples - 1, multiples])


def test_precomputed_multiplier_is_exact(monkeypatch):
    # The kernel runs on the host, so building it needs nvcc but no GPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    kernel = build_precompute_kernel()
    monkeypatch.undo()

    n = len(DIVISORS)
    m = torch.zeros(n, dtype=torch.uint32)
    s = torch.zeros(n, dtype=torch.uint32)
    kernel(m, s, torch.tensor(DIVISORS, dtype=torch.uint32), n)

    wrong = []
    for b, multiplier, shift in zip(DIVISORS, m.tolist(), s.tolist()):
        a = boundary_dividends(b)
        if not torch.equal(fastdiv_runtime(a, multiplier, shift), a // b):
            wrong.append(b)
    assert not wrong, "{} of {} divisors give a wrong quotient, the first are {}".format(
        len(wrong), len(DIVISORS), wrong[:8]
    )


class FastDivmodExample(tilus.Script):
    def __call__(self, base: int32, b: int32, quotient: ~int32, remainder: ~int32):
        self.attrs.blocks = (256, 1, 1)
        self.attrs.warps = 1

        q, r = self.fast_divmod(base + self.blockIdx.x, b)
        g_q = self.global_view(quotient, dtype=int32, shape=[256])
        g_r = self.global_view(remainder, dtype=int32, shape=[256])
        with self.thread_group(0, num_threads=1):
            self.store_global(
                g_q, self.register_tensor(dtype=int32, shape=[1], init=lambda i: q), offsets=[self.blockIdx.x]
            )
            self.store_global(
                g_r, self.register_tensor(dtype=int32, shape=[1], init=lambda i: r), offsets=[self.blockIdx.x]
            )


@requires.nvgpu_sm80
@pytest.mark.parametrize(
    "base, b",
    [
        (0, 3),
        (0, 7),
        (0, 56),
        (0, 64),
        (MAX_DIVIDEND - 255, 7),
    ],
)
def test_fast_divmod(base, b):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quotient = torch.full((256,), -1, dtype=torch.int32, device=device)
    remainder = torch.full((256,), -1, dtype=torch.int32, device=device)

    FastDivmodExample()(base, b, quotient, remainder)

    a = torch.arange(base, base + 256, dtype=torch.int64, device=device)
    torch.testing.assert_close(quotient.to(torch.int64), a // b)
    torch.testing.assert_close(remainder.to(torch.int64), a % b)
