# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tilus.ir.inst import Instruction


@dataclass(frozen=True, eq=False)
class SetMaxNRegInst(Instruction):
    """Adjust the per-thread register limit of the executing warp group: setmaxnreg.{action}."""

    nregs: int
    action: str

    @staticmethod
    def create(nregs: int, action: str) -> SetMaxNRegInst:
        if action not in ("inc", "dec"):
            raise ValueError(f"Invalid setmaxnreg action: {action}. Supported candidates are 'inc' and 'dec'.")
        if not (isinstance(nregs, int) and 24 <= nregs <= 256 and nregs % 8 == 0):
            raise ValueError(f"setmaxnreg count must be a multiple of 8 in [24, 256], got {nregs}.")
        return SetMaxNRegInst(output=None, inputs=(), nregs=nregs, action=action)
