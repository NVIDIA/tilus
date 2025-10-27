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
from typing import Literal

import pytest
import tilus
import torch
from tilus import int32, uint64
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

        barriers = self.shared_tensor(dtype=uint64, shape=[1])
        self.mbarrier.init(~barriers[0], count=1)
        self.sync()

        # load x from global to shared
        self.copy_async(src=g_x, dst=s_x, offsets=[m_offset, n_offset])
        self.copy_async_wait_all()
        self.sync()

        # copy x from shared to tmem
        self.tcgen05.copy(src=s_x, dst=t_x)
        self.tcgen05.commit(mbarrier=~barriers[0])
        self.mbarrier.wait(~barriers[0], phase=0)

        # load y from tmem to register
        r_y = self.tcgen05.load(t_x, offsets=[0, 0], shape=[self.block_m, self.block_n])
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


if __name__ == "__main__":
    pytest.main([__file__])
