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
from typing import Sequence

from hidet import uint32
from hidet.ir.expr import Expr

from tilus.ir.inst import Instruction
from tilus.ir.tensor import RegisterTensor


@dataclass(frozen=True, eq=False)
class AllocBarrierInst(Instruction):
    counts: tuple[Expr | None, ...]

    @staticmethod
    def create(counts: Sequence[Expr | None]) -> AllocBarrierInst:
        out = RegisterTensor.create(dtype=uint32, shape=[len(counts)], optional_layout=None)
        return AllocBarrierInst(output=out, inputs=(), counts=tuple(counts))


@dataclass(frozen=True, eq=False)
class ArriveBarrierInst(Instruction):
    barrier: Expr
    per_thread_count: Expr

    @staticmethod
    def create(barrier: Expr, per_thread_count: Expr) -> ArriveBarrierInst:
        return ArriveBarrierInst(output=None, inputs=(), barrier=barrier, per_thread_count=per_thread_count)


@dataclass(frozen=True, eq=False)
class WaitBarrierInst(Instruction):
    barrier: Expr
    phase: Expr

    @staticmethod
    def create(barrier: Expr, phase: Expr) -> WaitBarrierInst:
        return WaitBarrierInst(output=None, inputs=(), barrier=barrier, phase=phase)


@dataclass(frozen=True, eq=False)
class FenceProxyCopyAsync(Instruction):
    @staticmethod
    def create() -> FenceProxyCopyAsync:
        return FenceProxyCopyAsync(output=None, inputs=())
