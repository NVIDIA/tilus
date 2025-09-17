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
class CopyAsyncBulkGlobalToSharedInst(Instruction):
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
    ) -> CopyAsyncBulkGlobalToSharedInst:
        offsets_ = tuple(as_expr(offset) for offset in offsets)
        return CopyAsyncBulkGlobalToSharedInst(
            output=None,
            inputs=(dst, src),
            offsets=offsets_,
            mbarrier=mbarrier,
            dims=tuple(dims) if dims else None,
            evict=evict,
            check_bounds=check_bounds,
        )


@dataclass(frozen=True, eq=False)
class CopyAsyncBulkGlobalToClusterSharedInst(Instruction):
    offsets: tuple[Expr, ...]
    dims: tuple[int, ...]
    mbarrier: Expr
    cta_mask: int
    evict: Optional[str]
    check_bounds: bool = True

    @staticmethod
    def create(
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Sequence[int],
        mbarrier: Expr,
        cta_mask: int,
        evict: Optional[str] = None,
        check_bounds: bool = True,
    ) -> CopyAsyncBulkGlobalToClusterSharedInst:
        offsets_ = tuple(as_expr(offset) for offset in offsets)
        return CopyAsyncBulkGlobalToClusterSharedInst(
            output=None,
            inputs=(dst, src),
            offsets=offsets_,
            mbarrier=mbarrier,
            cta_mask=cta_mask,
            dims=tuple(dims) if dims else None,
            evict=evict,
            check_bounds=check_bounds,
        )


@dataclass(frozen=True, eq=False)
class CopyAsyncBulkSharedToGlobalInst(Instruction):
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
    ) -> CopyAsyncBulkSharedToGlobalInst:
        offsets_ = tuple(as_expr(offset) for offset in offsets)
        return CopyAsyncBulkSharedToGlobalInst(
            output=None,
            inputs=(dst, src),
            offsets=offsets_,
            dims=tuple(dims) if dims else None,
            l2_evict=l2_evict,
            check_bounds=check_bounds,
        )


@dataclass(frozen=True, eq=False)
class CopyAsyncBulkSharedToClusterSharedInst(Instruction):
    mbarrier: Expr
    remote_rank: int

    @staticmethod
    def create(
        src: SharedTensor,
        dst: SharedTensor,
        remote_rank: int,
        mbarrier: Expr,
    ) -> CopyAsyncBulkSharedToClusterSharedInst:
        return CopyAsyncBulkSharedToClusterSharedInst(
            output=None,
            inputs=(dst, src),
            remote_rank=remote_rank,
            mbarrier=mbarrier,
        )


@dataclass(frozen=True, eq=False)
class CopyAsyncBulkCommitGroupInst(Instruction):
    pass


@dataclass(frozen=True, eq=False)
class CopyAsyncBulkWaitGroupInst(Instruction):
    n: int

    @staticmethod
    def create(n: int) -> CopyAsyncBulkWaitGroupInst:
        return CopyAsyncBulkWaitGroupInst(output=None, inputs=(), n=n)

