# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Sequence

from tilus.hidet.ir.builders import StmtBuilder
from tilus.hidet.ir.dtypes import int32
from tilus.hidet.ir.expr import Expr, logical_and
from tilus.hidet.ir.stmt import Stmt
from tilus.hidet.lang.constructs.context import HidetContext


class WarpGroupContext(HidetContext):
    """
    A context manager for CUDA warp specialization, which allows different warps (groups of 32 threads)
    to perform different roles in producer-consumer patterns.

    Example usage:
    ```python
    with warp_groups([0, 1]) as tid
        # Code for warp groups 0 and 1 (warps 0-7)
        body1(tid)

    with warp_groups([2, 3]) as tid
        # Code for warp groups 2 and 3 (warps 8-15)
        body2(tid)
    ```

    And Hidet will make it semantically equivalent to the following code:
    ```python
    if threadIdx.x // 128 in [0, 1]:
        tid = threadIdx.x % 256
        body1(tid)

    if threadIdx.x // 128 in [2, 3]:
        tid = threadIdx.x % 256
        body2(tid)
    ```

    The warp groups must be contiguous and non-overlapping, and the warp group IDs must be unique.
    """

    def __init__(self, group_ids: Sequence[int]):
        """
        Initialize a warp specialization context.

        Args:
            group_ids: List of warp group IDs (each group = 4 warps = 128 threads) assigned to this role
        """
        assert len(set(group_ids)) == len(group_ids), "Duplicate warp group IDs are not allowed"
        assert len(set(group_ids)) == max(group_ids) - min(group_ids) + 1, "Warp group IDs must be contiguous"

        from tilus.hidet.lang.cuda import threadIdx

        min_warpgroup_id: int = min(group_ids)
        max_warpgroup_id: int = max(group_ids)
        num_warpgroups: int = max_warpgroup_id - min_warpgroup_id + 1

        self.condition: Expr = logical_and(
            min_warpgroup_id * 128 <= threadIdx.x, threadIdx.x < (max_warpgroup_id + 1) * 128
        )
        self.tid_value = (threadIdx.x - int32(min_warpgroup_id * 128)) % int32(num_warpgroups * 128)

    def bind_value(self) -> Expr:
        return self.tid_value

    def post_process(self, stmt: Stmt) -> Stmt:
        sb = StmtBuilder()

        with sb.if_then(self.condition):
            sb.append(stmt)

        return sb.finish()


def warp_groups(group_ids: Sequence[int]) -> WarpGroupContext:
    return WarpGroupContext(group_ids)
