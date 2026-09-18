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
from tilus.ir.instructions.cuda.tcgen05 import Tcgen05AllocInst
from tilus.ir.layout import TMemoryLayout
from tilus.ir.layout.inference.rule import (
    LayoutInferenceContext,
    LayoutInferenceRule,
    register_rule,
)
from tilus.ir.layout.ops.tmemory_ops import tmemory_row_major
from tilus.ir.layout.tmem_layout import TMemoryDuplication
from tilus.ir.tensor import TMemoryTensor

# Map from shape[0] -> duplication mode that's uniquely determined by the lane size.
# shape[0]=64 is ambiguous (NONE / WARPX2_02_13 / WARPX2_01_23) and is left for
# downstream inference (e.g. from a tcgen05.copy multicast hint).
_LANE_TO_FORCED_DUPLICATION: dict[int, TMemoryDuplication] = {
    32: TMemoryDuplication.WARPX4,
    128: TMemoryDuplication.NONE,
}


@register_rule(Tcgen05AllocInst)
class Tcgen05AllocRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: Tcgen05AllocInst) -> dict[TMemoryTensor, TMemoryLayout]:
        tmem = inst.tmemory_output
        if tmem.optional_layout is not None:
            return {}
        lane_size = tmem.shape[0]
        if lane_size not in _LANE_TO_FORCED_DUPLICATION:
            # shape[0]=64 — defer to downstream inference (writers/consumers).
            return {}
        duplication = _LANE_TO_FORCED_DUPLICATION[lane_size]
        return {tmem: tmemory_row_major(tmem.shape, duplication=duplication)}
