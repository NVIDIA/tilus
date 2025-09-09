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
class BulkCopyAsyncGlobalToSharedInst(Instruction):
    offsets: tuple[Expr, ...]
    dims: tuple[int, ...]
    mbarrier: Expr
    evict: Optional[str]
    check_bounds: bool = True

    @staticmethod
    def create(
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Sequence[int],
        mbarrier: Expr,
        evict: Optional[str] = None,
        check_bounds: bool = True,
    ) -> BulkCopyAsyncGlobalToSharedInst:
        offsets_ = tuple(as_expr(offset) for offset in offsets)
        return BulkCopyAsyncGlobalToSharedInst(
            output=None,
            inputs=(dst, src),
            offsets=offsets_,
            mbarrier=mbarrier,
            dims=tuple(dims) if dims else None,
            evict=evict,
            check_bounds=check_bounds,
        )


@dataclass(frozen=True, eq=False)
class BulkCopyAsyncGlobalToClusterSharedInst(Instruction):
    offsets: tuple[Expr, ...]
    dims: tuple[int, ...]
    mbarrier: Expr
    evict: Optional[str]
    check_bounds: bool = True

    @staticmethod
    def create(
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Sequence[int],
        mbarrier: Expr,
        evict: Optional[str] = None,
        check_bounds: bool = True,
    ) -> BulkCopyAsyncGlobalToClusterSharedInst:
        offsets_ = tuple(as_expr(offset) for offset in offsets)
        return BulkCopyAsyncGlobalToClusterSharedInst(
            output=None,
            inputs=(dst, src),
            offsets=offsets_,
            mbarrier=mbarrier,
            dims=tuple(dims) if dims else None,
            evict=evict,
            check_bounds=check_bounds,
        )


@dataclass(frozen=True, eq=False)
class BulkCopyAsyncSharedToGlobalInst(Instruction):
    offsets: tuple[Expr, ...]
    dims: tuple[int, ...]
    check_bounds: bool = True
    l2_evict: Optional[str] = "evict_first"

    @staticmethod
    def create(
        src: SharedTensor,
        dst: GlobalTensor,
        offsets: Sequence[Expr | int],
        dims: Sequence[int],
        l2_evict: Optional[str] = "evict_first",
        check_bounds: bool = True,
    ) -> BulkCopyAsyncSharedToGlobalInst:
        offsets_ = tuple(as_expr(offset) for offset in offsets)
        return BulkCopyAsyncSharedToGlobalInst(
            output=None,
            inputs=(dst, src),
            offsets=offsets_,
            dims=tuple(dims) if dims else None,
            l2_evict=l2_evict,
            check_bounds=check_bounds,
        )


@dataclass(frozen=True, eq=False)
class BulkCopyAsyncSharedToClusterSharedInst(Instruction):
    mbarrier: Expr

    @staticmethod
    def create(
        src: SharedTensor,
        dst: SharedTensor,
        mbarrier: Expr,
    ) -> BulkCopyAsyncSharedToClusterSharedInst:
        return BulkCopyAsyncSharedToClusterSharedInst(
            output=None,
            inputs=(dst, src),
            mbarrier=mbarrier,
        )
