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
from typing import Literal

import pytest
import tilus
import tilus.testing
import torch
from tilus import int32
from tilus.ir.layout.cuda.tcgen05.smem import (
    Tcgen05SwizzleMode,
    generate_canonical_layout,
)
from tilus.utils import cdiv


class TmemCopyExample(tilus.Script):
    def __init__(self, major_kind: Literal["MN", "K"], swizzle_mode: Tcgen05SwizzleMode):
        super().__init__()
        self.block_m = 128
        self.block_n = 32
        self.shared_layout = generate_canonical_layout(
            shape=(self.block_m, self.block_n),
            dtype=int32,
            major_kind=major_kind,
            swizzle_mode=swizzle_mode,
        ).as_shared_layout()

    def __call__(self, m_size: int, n_size: int, x_ptr: ~int32, y_ptr: ~int32):
        self.attrs.blocks = cdiv(m_size, self.block_m), cdiv(n_size, self.block_n)
        self.attrs.warps = 4

        m_offset = self.blockIdx.x * self.block_m
        n_offset = self.blockIdx.y * self.block_n

        g_x = self.global_view(x_ptr, dtype=int32, shape=[m_size, n_size])
        g_y = self.global_view(y_ptr, dtype=int32, shape=[m_size, n_size])

        s_x = self.shared_tensor(dtype=int32, shape=[self.block_m, self.block_n])
        t_x = self.tcgen05.alloc(dtype=int32, shape=[self.block_m, self.block_n])

        barriers = self.mbarrier.alloc(counts=[1])

        # load x from global to shared
        self.copy_async(src=g_x, dst=s_x, offsets=[m_offset, n_offset])
        self.copy_async_wait_all()
        self.sync()

        # copy x from shared to tmem
        with self.single_warp():
            self.tcgen05.copy(src=s_x, dst=t_x)
            self.tcgen05.commit(mbarrier=barriers[0])
        self.mbarrier.wait(barriers[0], phase=0)

        # load y from tmem to register
        r_y = self.tcgen05.load(t_x)
        self.tcgen05.wait_load()
        self.sync()

        # store y from register to global
        self.store_global(g_y, r_y, offsets=[m_offset, n_offset])

        self.tcgen05.dealloc(t_x)

        self.annotate_layout(s_x, self.shared_layout)


@tilus.testing.requires.nvgpu_sm100a
@pytest.mark.parametrize(
    "major_kind, swizzle_mode",
    [
        ("MN", Tcgen05SwizzleMode.NO_SWIZZLE),
        ("MN", Tcgen05SwizzleMode.B32_SWIZZLE),
        ("MN", Tcgen05SwizzleMode.B64_SWIZZLE),
        ("MN", Tcgen05SwizzleMode.B128_SWIZZLE),
        ("K", Tcgen05SwizzleMode.NO_SWIZZLE),
        ("K", Tcgen05SwizzleMode.B32_SWIZZLE),
        ("K", Tcgen05SwizzleMode.B64_SWIZZLE),
        ("K", Tcgen05SwizzleMode.B128_SWIZZLE),
    ],
)
def test_tcgen05_copy(major_kind, swizzle_mode):
    if major_kind == "MN":
        pytest.xfail("MN is not supported")
    m_size = 128
    n_size = 32
    x = torch.randint(0, 128, [m_size, n_size], dtype=torch.int32, device="cuda")
    y = torch.ones([m_size, n_size], dtype=torch.int32, device="cuda")
    kernel = TmemCopyExample(major_kind=major_kind, swizzle_mode=swizzle_mode)
    kernel(m_size, n_size, x, y)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, y)


class TmemCopyMulticastExample(tilus.Script):
    """Round-trip SMEM -> TMEM (multicast) -> register -> global.

    Warps are chosen so the read-back covers the unique source rows of each
    multicast. Multicasts replicate the source across TMEM sub-partitions; the load
    instruction is constrained to ``current_num_threads == tmem.shape[0]``.
    Reading exactly ``shape[0]`` rows back lets us verify the data the kernel
    placed into TMEM. For ``warpx2_01_23`` the two warp pairs receive duplicate
    halves, so the expected output is a tiling of the first 32 source rows.
    """

    def __init__(self, multicast: str, num_warps: int, block_m: int):
        super().__init__()
        self.multicast = multicast
        self.num_warps = num_warps
        self.block_m = block_m
        self.block_n = 32
        self.shared_layout = generate_canonical_layout(
            shape=(self.block_m, self.block_n),
            dtype=int32,
            major_kind="K",
            swizzle_mode=Tcgen05SwizzleMode.NO_SWIZZLE,
        ).as_shared_layout()

    def __call__(self, m_size: int, n_size: int, x_ptr: ~int32, y_ptr: ~int32):
        self.attrs.blocks = cdiv(m_size, self.block_m), cdiv(n_size, self.block_n)
        self.attrs.warps = self.num_warps

        m_offset = self.blockIdx.x * self.block_m
        n_offset = self.blockIdx.y * self.block_n

        g_x = self.global_view(x_ptr, dtype=int32, shape=[m_size, n_size])
        g_y = self.global_view(y_ptr, dtype=int32, shape=[m_size, n_size])

        s_x = self.shared_tensor(dtype=int32, shape=[self.block_m, self.block_n])
        t_x = self.tcgen05.alloc(dtype=int32, shape=[self.block_m, self.block_n])

        barriers = self.mbarrier.alloc(counts=[1])

        self.copy_async(src=g_x, dst=s_x, offsets=[m_offset, n_offset])
        self.copy_async_wait_all()
        self.sync()

        with self.single_warp():
            self.tcgen05.copy(src=s_x, dst=t_x, multicast=self.multicast)
            self.tcgen05.commit(mbarrier=barriers[0])
        self.mbarrier.wait(barriers[0], phase=0)

        r_y = self.tcgen05.load(t_x)
        self.tcgen05.wait_load()
        self.sync()

        self.store_global(g_y, r_y, offsets=[m_offset, n_offset])

        self.tcgen05.dealloc(t_x)

        self.annotate_layout(s_x, self.shared_layout)


@tilus.testing.requires.nvgpu_sm100a
@pytest.mark.parametrize(
    "multicast, num_warps, block_m",
    [
        ("warpx4", 1, 32),
        ("warpx2_02_13", 2, 64),
        ("warpx2_01_23", 2, 64),
    ],
)
def test_tcgen05_copy_multicast(multicast: str, num_warps: int, block_m: int):
    n_size = 32
    x = torch.randint(0, 128, [block_m, n_size], dtype=torch.int32, device="cuda")
    y = torch.zeros([block_m, n_size], dtype=torch.int32, device="cuda")
    kernel = TmemCopyMulticastExample(multicast=multicast, num_warps=num_warps, block_m=block_m)
    kernel(block_m, n_size, x, y)
    torch.cuda.synchronize()
    if multicast == "warpx2_01_23":
        # Warps 0 and 1 share the first 32 source rows; the loaded TMEM
        # therefore tiles src[0:32] across both halves of the output.
        expected = torch.cat([x[:32], x[:32]], dim=0)
    else:
        expected = x
    torch.testing.assert_close(y, expected)


def test_tcgen05_copy_multicast_invalid_name():
    """Reject unknown multicast names with a clear ``Unknown multicast`` error.

    The lang-layer ``copy(..., multicast=...)`` validates the string against
    the allowed multicast-name set.
    """

    class _BadMulticast(tilus.Script):
        def __init__(self):
            super().__init__()

        def __call__(self, x_ptr: ~int32, y_ptr: ~int32):
            self.attrs.blocks = 1
            self.attrs.warps = 4
            g_x = self.global_view(x_ptr, dtype=int32, shape=[128, 32])
            s_x = self.shared_tensor(dtype=int32, shape=[128, 32])
            t_x = self.tcgen05.alloc(dtype=int32, shape=[128, 32])
            self.copy_async(src=g_x, dst=s_x, offsets=[0, 0])
            self.copy_async_wait_all()
            self.sync()
            with self.single_warp():
                self.tcgen05.copy(src=s_x, dst=t_x, multicast="warpx_bogus")
            self.tcgen05.dealloc(t_x)

    kernel = _BadMulticast()
    x = torch.zeros(128, 32, dtype=torch.int32, device="cuda")
    y = torch.zeros(128, 32, dtype=torch.int32, device="cuda")
    with pytest.raises(Exception, match="Unknown multicast"):
        kernel(x, y)


if __name__ == "__main__":
    pytest.main([__file__])
