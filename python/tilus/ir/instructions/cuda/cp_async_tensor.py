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
    evict: Optional[str]

    @staticmethod
    def create(
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Sequence[int],
        mbarrier: Expr,
        evict: Optional[str] = None,
    ) -> CopyAsyncTensorGlobalToSharedInst:
        offsets_ = tuple(as_expr(offset) for offset in offsets)
        return CopyAsyncTensorGlobalToSharedInst(
            output=None,
            inputs=(dst, src),
            offsets=offsets_,
            mbarrier=mbarrier,
            dims=tuple(dims) if dims else None,
            evict=evict,
        )

@dataclass(frozen=True, eq=False)
class CopyAsyncTensorSharedToGlobalInst(Instruction):
    offsets: tuple[Expr, ...]
    dims: tuple[int, ...]
    evict: Optional[str]

    @staticmethod
    def create(
        src: SharedTensor,
        dst: GlobalTensor,
        offsets: Sequence[Expr | int],
        dims: Sequence[int],
        evict: Optional[str] = None,
    ) -> CopyAsyncTensorSharedToGlobalInst:
        offsets_ = tuple(as_expr(offset) for offset in offsets)
        return  CopyAsyncTensorSharedToGlobalInst(
            output=None,
            inputs=(dst, src),
            offsets=offsets_,
            dims=tuple(dims) if dims else None,
            evict=evict,
        )
