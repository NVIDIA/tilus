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
import tilus
import torch
from tilus import float32, int32
from tilus.ir.layout import RegisterLayout
from tilus.ir.layout.ops import spatial
from tilus.utils import cdiv


class MemoryCopy(tilus.Script):
    def __init__(self, num_warps: int = 4):
        super().__init__()
        self.num_warps: int = num_warps
        self.block_size: int = num_warps * 32
        self.layout: RegisterLayout = spatial(num_warps * 32)

    def __call__(self, n: int32, src_ptr: ~float32, dst_ptr: ~float32):  # type: ignore
        self.attrs.blocks = [cdiv(n, self.block_size) * self.block_size]
        self.attrs.warps = self.num_warps

        bi = self.blockIdx.x

        loaded_regs = self.load_global_generic(
            dtype=float32,
            shape=self.layout.shape,
            layout=self.layout,
            ptr=src_ptr,
            f_offset=lambda i: bi * self.block_size + i,
            f_mask=lambda i: bi * self.block_size + i < n,
        )
        self.store_global_generic(
            loaded_regs,
            ptr=dst_ptr,
            f_offset=lambda i: bi * self.block_size + i,
            f_mask=lambda i: bi * self.block_size + i < n,
        )


def test_tilus_script_with_copy_example():
    a = torch.ones(12, dtype=torch.float32).cuda()
    b = torch.empty(12, dtype=torch.float32).cuda()

    script = MemoryCopy()
    n = a.size(0)

    # launch the kernel
    script(n, a, b)

    torch.testing.assert_close(a, b)
