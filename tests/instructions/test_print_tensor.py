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
from tilus import float32
from tilus.testing import requires


class PrintTmemTensor(tilus.Script):
    def __call__(self):
        self.attrs.blocks = 1
        self.attrs.warps = 4

        t_a = self.tcgen05.alloc(dtype=float32, shape=[128, 32])
        self.print_tensor("t_a: ", t_a)


@requires.nvgpu_sm100a
def test_print_tmem_tensor():
    kernel = PrintTmemTensor()
    kernel()
