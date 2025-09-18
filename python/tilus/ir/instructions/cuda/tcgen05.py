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

from hidet.ir.dtypes import uint32

from tilus.ir.inst import Instruction
from tilus.ir.tensor import TensorMemoryTensor


@dataclass(frozen=True, eq=False)
class Tcgen05AllocInst(Instruction):
    cta_group: int  # 1 or 2

    @staticmethod
    def create(num_columns: int, cta_group: int) -> Tcgen05AllocInst:
        assert cta_group in (1, 2)
        output = TensorMemoryTensor.create(dtype=uint32, shape=[128, num_columns])
        return Tcgen05AllocInst(output=output, inputs=(), cta_group=cta_group)


@dataclass(frozen=True, eq=False)
class Tcgen05DeallocInst(Instruction):
    @staticmethod
    def create(tmt: TensorMemoryTensor) -> Tcgen05DeallocInst:
        return Tcgen05DeallocInst(output=None, inputs=(tmt,))


@dataclass(frozen=True, eq=False)
class Tcgen05RelinquishAllocPermitInst(Instruction):
    cta_group: int = 1

    @staticmethod
    def create(cta_group: int) -> Tcgen05RelinquishAllocPermitInst:
        return Tcgen05RelinquishAllocPermitInst(output=None, inputs=(), cta_group=cta_group)
