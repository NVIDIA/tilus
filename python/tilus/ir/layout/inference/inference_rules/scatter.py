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
"""Layout inference for scatter instructions (atomic and non-atomic).

All four variants (``AtomicScatterSharedInst``, ``AtomicScatterGlobalInst``,
``StoreSharedScatterInst``, ``StoreGlobalScatterInst``) need the same rule:
``indices``, ``values``, and the optional ``output`` must share a single
RegisterLayout so each lane reads its own element at the same local index
across all three tensors.

The rule picks ``auto_local_spatial`` over the ``indices`` shape as a
reasonable default; downstream analyses or a user-supplied ``annotate_layout``
can still pin a different layout.
"""

from __future__ import annotations

from tilus.ir.instructions import (
    AtomicScatterGlobalInst,
    AtomicScatterSharedInst,
    StoreGlobalScatterInst,
    StoreSharedScatterInst,
)
from tilus.ir.layout import RegisterLayout
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.layout.ops.register_ops import auto_local_spatial
from tilus.ir.tensor import RegisterTensor


def _choose_layout(tensor: RegisterTensor, num_threads: int) -> RegisterLayout:
    if tensor.optional_layout:
        return tensor.optional_layout
    return auto_local_spatial(num_threads=num_threads, shape=list(tensor.shape))


@register_rule(AtomicScatterSharedInst)
@register_rule(AtomicScatterGlobalInst)
@register_rule(StoreSharedScatterInst)
@register_rule(StoreGlobalScatterInst)
class ScatterRule(LayoutInferenceRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext,
        inst: (AtomicScatterSharedInst | AtomicScatterGlobalInst | StoreSharedScatterInst | StoreGlobalScatterInst),
    ) -> dict[RegisterTensor, RegisterLayout]:
        indices = inst.inputs[1].as_register_tensor()
        values = inst.inputs[2].as_register_tensor()
        layout = _choose_layout(indices, ctx.num_threads)
        out: dict[RegisterTensor, RegisterLayout] = {indices: layout, values: layout}
        # Atomic scatter variants carry an optional output; non-atomic store
        # variants do not.
        if getattr(inst, "output", None) is not None:
            out[inst.register_output] = layout
        return out
