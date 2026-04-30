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
"""Layout inference and validation rules for Philox4x32Inst.

The output layout must be ``local(4, 1, ..., 1) * offset_layout`` — i.e., the leading
dimension of size 4 is purely local (each thread holds all 4 Philox outputs), and the
remaining dimensions match the offset layout exactly.
"""

from tilus.ir.instructions import Philox4x32Inst
from tilus.ir.layout import RegisterLayout, ops
from tilus.ir.layout.inference.rule import (
    LayoutInferenceContext,
    LayoutInferenceRule,
    LayoutValidationRule,
    register_rule,
)
from tilus.ir.tensor import RegisterTensor


def _output_layout_from_offset(offset_layout: RegisterLayout) -> RegisterLayout:
    """Compute the expected output layout: local(4) composed with offset_layout via unsqueeze."""
    # Unsqueeze adds a dim-0 of size 1, then compose with local(4) to get size 4 in dim 0
    unsqueezed = ops.unsqueeze(offset_layout, dims=[0])
    local_4 = ops.local(4, *([1] * len(offset_layout.shape)))
    return ops.compose(local_4, unsqueezed)


def _offset_layout_from_output(output_layout: RegisterLayout) -> RegisterLayout:
    """Extract offset layout from output layout by squeezing the leading dim-4."""
    # Reduce dim 0 (size 4) then squeeze it out
    reduced = ops.reduce(output_layout, dims=[0], keepdims=False)
    return reduced


@register_rule(Philox4x32Inst)
class Philox4x32InferenceRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: Philox4x32Inst) -> dict[RegisterTensor, RegisterLayout]:
        offset = inst.inputs[0].as_register_tensor()
        output = inst.register_output

        if offset.optional_layout is not None and output.optional_layout is not None:
            return {}
        elif offset.optional_layout is not None:
            return {output: _output_layout_from_offset(offset.layout)}
        elif output.optional_layout is not None:
            return {offset: _offset_layout_from_output(output.layout)}
        else:
            return {}


@register_rule(Philox4x32Inst)
class Philox4x32ValidationRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: Philox4x32Inst) -> bool:
        offset = inst.inputs[0].as_register_tensor()
        output = inst.register_output

        if offset.optional_layout is None or output.optional_layout is None:
            return True

        expected_output = _output_layout_from_offset(offset.layout)
        return output.layout == expected_output
