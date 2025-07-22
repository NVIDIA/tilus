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
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tilus.ir.inst import Instruction
from tilus.ir.tensor import RegisterTensor


@dataclass(frozen=True, eq=False)
class SimtDotInst(Instruction):
    warp_spatial: tuple[int, int, int]
    warp_repeat: tuple[int, int, int]
    thread_spatial: tuple[int, int]
    thread_repeat: tuple[int, int]

    @staticmethod
    def create(
        a: RegisterTensor,
        b: RegisterTensor,
        c: RegisterTensor,
        warp_spatial: tuple[int, int, int],
        warp_repeat: tuple[int, int, int],
        thread_spatial: tuple[int, int],
        thread_repeat: tuple[int, int],
        output: Optional[RegisterTensor] = None,
    ) -> SimtDotInst:
        if output is None:
            output = RegisterTensor.create(c.dtype, shape=c.shape, optional_layout=c.optional_layout)
        return SimtDotInst(
            output=output,
            inputs=(a, b, c),
            warp_spatial=warp_spatial,
            warp_repeat=warp_repeat,
            thread_spatial=thread_spatial,
            thread_repeat=thread_repeat,
        )
