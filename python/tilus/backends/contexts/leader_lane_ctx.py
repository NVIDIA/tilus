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

from typing import Optional

from tilus.backends.context import BaseEmitContext
from tilus.hidet.ir.expr import Var


class LeaderLaneContext(BaseEmitContext):
    """Context that manages a per-warp leader lane predicate for warp-cooperative instructions.

    Many SASS instructions (UTMASTG, UTCMMA, CLC.TRY_CANCEL, TMA copy) are warp-cooperative:
    all threads in a warp participate in their execution, but the PTX semantics require them to
    be issued by a single thread. When emitting through PTX, wrapping these instructions in
    an if-branch (via elect_sync) causes ptxas to generate BSSY/BSYNC divergence overhead.

    This context provides a pre-computed `is_leader_lane` uint32 variable (1 for one elected
    thread per warp, 0 for others) that can be passed as a predicate directly into PTX inline
    asm via `@p` syntax, avoiding the if-branch entirely.

    The variable is lazily initialized on first access and declared at the kernel's outermost scope.
    """

    def __post_init__(self):
        self._leader_lane_var: Optional[Var] = None

    @property
    def leader_lane(self) -> Var:
        """Get or lazily create the per-warp leader lane predicate variable.

        Returns a Var reference. The actual declaration is emitted in finalize().

        Returns
        -------
        Var
            A uint32 variable: 1 for the elected leader thread in each warp, 0 for all others.
        """
        if self._leader_lane_var is None:
            from tilus.hidet.ir.dtypes import uint32

            self._leader_lane_var = Var("is_leader_lane", type=uint32)
        return self._leader_lane_var

    def finalize(self):
        """Emit the leader lane declaration at the kernel's outermost scope if it was accessed."""
        if self._leader_lane_var is not None:
            from tilus.hidet.ir.primitives.cuda.elect import elect_one_sync
            from tilus.hidet.ir.stmt import DeclareStmt

            self.kernel_prepend(DeclareStmt(self._leader_lane_var, init=elect_one_sync()))
