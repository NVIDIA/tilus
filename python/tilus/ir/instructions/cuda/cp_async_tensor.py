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
from typing import Optional, Sequence

from hidet.ir.expr import Expr, as_expr

from tilus.ir.inst import Instruction
from tilus.ir.tensor import GlobalTensor, SharedTensor


@dataclass(frozen=True, eq=False)
class CopyAsyncTensorGlobalToSharedInst(Instruction):
    offsets: tuple[Expr, ...]
    dims: tuple[int, ...]
    mbarrier: Expr
    cache_policy: Optional[Expr]

    @staticmethod
    def create(
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Sequence[int],
        mbarrier: Expr,
        cache_policy: Optional[Expr] = None
    ) -> CopyAsyncTensorGlobalToSharedInst:
        offsets_ = tuple(as_expr(offset) for offset in offsets)
        return CopyAsyncTensorGlobalToSharedInst(
            output=None,
            inputs=(dst, src),
            offsets=offsets_,
            mbarrier=mbarrier,
            dims=tuple(dims) if dims else None,
            cache_policy=cache_policy
        )

@dataclass(frozen=True, eq=False)
class CopyAsyncTensorSharedToGlobalInst(Instruction):
    offsets: tuple[Expr, ...]
    dims: tuple[int, ...]
    cache_policy: Optional[Expr]

    @staticmethod
    def create(
        src: SharedTensor,
        dst: GlobalTensor,
        offsets: Sequence[Expr | int],
        dims: Sequence[int],
        cache_policy: Optional[Expr] = None
    ) -> CopyAsyncTensorSharedToGlobalInst:
        offsets_ = tuple(as_expr(offset) for offset in offsets)
        return  CopyAsyncTensorSharedToGlobalInst(
            output=None,
            inputs=(dst, src),
            offsets=offsets_,
            dims=tuple(dims) if dims else None,
            cache_policy=cache_policy
        )

@dataclass(frozen=True, eq=False)
class CopyAsyncTensorCommitGroupInst(Instruction):
    @staticmethod
    def create() -> CopyAsyncTensorCommitGroupInst:
        return CopyAsyncTensorCommitGroupInst(output=None, inputs=())

@dataclass(frozen=True, eq=False)
class CopyAsyncTensorWaitGroupInst(Instruction):
    n: int

    @staticmethod
    def create(n: int) -> CopyAsyncTensorWaitGroupInst:
        return CopyAsyncTensorWaitGroupInst(output=None, inputs=(), n=n)
