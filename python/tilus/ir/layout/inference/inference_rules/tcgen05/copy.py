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
from tilus.ir.instructions.cuda.tcgen05 import Tcgen05CopyInst
from tilus.ir.layout import SharedLayout, TMemoryLayout
from tilus.ir.layout.cuda.tcgen05.smem import (
    Tcgen05SwizzleMode,
    generate_canonical_layout,
    get_shared_layout_from_canonical,
)
from tilus.ir.layout.inference.rule import (
    LayoutInferenceContext,
    LayoutInferenceError,
    LayoutInferenceRule,
    register_rule,
)
from tilus.ir.layout.ops.tmemory_ops import tmemory_row_major
from tilus.ir.layout.tmem_layout import TMemoryDuplication
from tilus.ir.tensor import SharedTensor, TMemoryTensor

# Map the user-facing multicast string on Tcgen05CopyInst to the corresponding
# TMEM duplication mode. Empty string means no multicast (no replication).
_MULTICAST_TO_DUPLICATION: dict[str, TMemoryDuplication] = {
    "": TMemoryDuplication.NONE,
    "warpx4": TMemoryDuplication.WARPX4,
    "warpx2_02_13": TMemoryDuplication.WARPX2_02_13,
    "warpx2_01_23": TMemoryDuplication.WARPX2_01_23,
}


@register_rule(Tcgen05CopyInst)
class Tcgen05CopyRule(LayoutInferenceRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: Tcgen05CopyInst
    ) -> dict[SharedTensor | TMemoryTensor, SharedLayout | TMemoryLayout]:
        result: dict[SharedTensor | TMemoryTensor, SharedLayout | TMemoryLayout] = {}
        dst = inst.inputs[0].as_tmemory_tensor()
        src = inst.inputs[1].as_shared_tensor()

        if not src.has_layout():
            if len(src.shape) != 2:
                raise LayoutInferenceError(f"Only 2D SharedTensor is supported in copy, got shape {src.shape}")
            canonical_layout = generate_canonical_layout(
                (src.shape[0], src.shape[1]), src.dtype, "K", Tcgen05SwizzleMode.NO_SWIZZLE
            )
            result[src] = get_shared_layout_from_canonical(canonical_layout)

        if not dst.has_layout():
            # Use the multicast kwarg to pick the right duplication. The
            # Tcgen05AllocRule already pins layouts for unambiguous shape[0]
            # (32 -> WARPX4, 128 -> NONE) and defers shape[0]=64 to here.
            duplication = _MULTICAST_TO_DUPLICATION.get(inst.multicast, TMemoryDuplication.NONE)
            result[dst] = tmemory_row_major(dst.shape, duplication=duplication)

        return result
