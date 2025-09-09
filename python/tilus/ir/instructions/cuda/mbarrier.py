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

from hidet.ir.expr import Expr

from tilus.ir.inst import Instruction


@dataclass(frozen=True, eq=False)
class InitBarrierInst(Instruction):
    barrier: Expr
    count: Optional[Expr]

    @staticmethod
    def create(barrier: Expr, count: Optional[Expr]) -> InitBarrierInst:
        return InitBarrierInst(output=None, inputs=(), barrier=barrier, count=count)


@dataclass(frozen=True, eq=False)
class ArriveBarrierInst(Instruction):
    barrier: Expr

    @staticmethod
    def create(barrier: Expr) -> ArriveBarrierInst:
        return ArriveBarrierInst(output=None, inputs=(), barrier=barrier)


@dataclass(frozen=True, eq=False)
class ArriveRemoteBarrierInst(Instruction):
    barrier: Expr
    remote_block: Expr

    @staticmethod
    def create(barrier: Expr, remote_block: Expr) -> ArriveRemoteBarrierInst:
        return ArriveRemoteBarrierInst(output=None, inputs=(), barrier=barrier, remote_block=remote_block)


@dataclass(frozen=True, eq=False)
class WaitBarrierInst(Instruction):
    barrier: Expr
    phase: Expr

    @staticmethod
    def create(barrier: Expr, phase: Expr) -> WaitBarrierInst:
        return WaitBarrierInst(output=None, inputs=(), barrier=barrier, phase=phase)
