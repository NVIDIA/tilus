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

from hidet.ir.dtypes import bf16, f8e4m3, f8e5m2, f16, i8, tf32, u1, u8
from hidet.ir.expr import Expr
from hidet.ir.type import DataType

from tilus.ir.inst import Instruction
from tilus.ir.tensor import RegisterTensor, SharedTensor
from tilus.utils import gcd


@dataclass(frozen=True, eq=False)
class WgmmaFenceInst(Instruction):
    @staticmethod
    def create() -> WgmmaFenceInst:
        return WgmmaFenceInst(output=None, inputs=())


@dataclass(frozen=True, eq=False)
class WgmmaCommitGroupInst(Instruction):
    @staticmethod
    def create() -> WgmmaCommitGroupInst:
        return WgmmaCommitGroupInst(output=None, inputs=())


@dataclass(frozen=True, eq=False)
class WgmmaWaitGroupInst(Instruction):
    n: Expr

    @staticmethod
    def create(n: Expr) -> WgmmaWaitGroupInst:
        return WgmmaWaitGroupInst(output=None, inputs=(), n=n)


@dataclass(frozen=True, eq=False)
class WgmmaMmaSSInst(Instruction):
    @staticmethod
    def get_inst_mnk(
        m: int, n: int, k: int, a_dtype: DataType, b_dtype: DataType, d_dtype: DataType
    ) -> tuple[int, int, int]:
        inst_m = 64
        inst_n = gcd(n, 256)  # why?
        if a_dtype == b_dtype == f16:
            inst_k = 16
        elif a_dtype == b_dtype == bf16:
            inst_k = 16
        elif a_dtype == b_dtype == tf32:
            inst_k = 8
        elif a_dtype in (f8e4m3, f8e5m2) and b_dtype in (f8e4m3, f8e5m2):
            inst_k = 32
        elif a_dtype in (i8, u8) and b_dtype in (i8, u8):
            inst_k = 32
        elif a_dtype == d_dtype == u1:
            inst_k = 256
        else:
            raise ValueError(f"Unsupported data types for MMA: a_dtype={a_dtype}, b_dtype={b_dtype}")
        return inst_m, inst_n, inst_k

    @staticmethod
    def create(a: SharedTensor, b: SharedTensor, d: RegisterTensor) -> WgmmaMmaSSInst:
        return WgmmaMmaSSInst(output=None, inputs=(a, b, d))


@dataclass(frozen=True, eq=False)
class WgmmaMmaRSInst(Instruction):
    @staticmethod
    def create(a: RegisterTensor, b: SharedTensor, d: RegisterTensor) -> WgmmaMmaRSInst:
        return WgmmaMmaRSInst(output=None, inputs=(a, b, d))
