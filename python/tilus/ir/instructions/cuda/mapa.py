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
from hidet.ir.expr import Expr

from tilus.ir.inst import Instruction, InstructionError
from tilus.ir.tensor import RegisterTensor


@dataclass(frozen=True, eq=False)
class MapSharedAddrInst(Instruction):
    target_rank: Expr

    @staticmethod
    def create(addr: RegisterTensor, target_rank: Expr) -> MapSharedAddrInst:
        if not isinstance(addr, RegisterTensor):
            raise InstructionError("addr must be a RegisterTensor, got {}".format(type(addr)))
        if addr.dtype != uint32:
            raise InstructionError("addr must have dtype uint32, got {}".format(addr.dtype))
        output = RegisterTensor(dtype=uint32, shape=addr.shape)
        return MapSharedAddrInst(output=output, inputs=(addr,), target_rank=target_rank)
