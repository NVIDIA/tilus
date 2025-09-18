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


class Tcgen05Example(tilus.Script):
    def __call__(self):
        self.attrs.blocks = 1
        self.attrs.warps = 4
        t_a = self.tcgen05.alloc(num_columns=32, cta_group=1)
        self.tcgen05.dealloc(t_a)
        self.tcgen05.relinquish_alloc_permit(cta_group=1)


@tilus.testing.requires.nvgpu_sm100
def test_tcgen_alloc_free():
    kernel = Tcgen05Example()
    kernel()
    torch.cuda.synchronize()
