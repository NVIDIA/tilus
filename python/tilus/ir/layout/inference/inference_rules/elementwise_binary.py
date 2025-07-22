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
from hidet.utils import same_list

from tilus.ir.instructions import AddInst, DivInst, ElementwiseBinaryInst, Instruction, ModInst, MulInst, SubInst
from tilus.ir.layout import RegisterLayout
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(AddInst)
@register_rule(SubInst)
@register_rule(DivInst)
@register_rule(MulInst)
@register_rule(ModInst)
@register_rule(ElementwiseBinaryInst)
class BinaryRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: Instruction) -> dict[RegisterTensor, RegisterLayout]:
        a = inst.inputs[0].as_register_tensor()
        b = inst.inputs[1].as_register_tensor()
        c = inst.output.as_register_tensor()

        if all(tensor.optional_layout is None for tensor in (a, b, c)):
            return {}
        elif c.optional_layout is not None:
            # c => a | b
            mapping = {}
            if a.optional_layout is None:
                mapping[a] = c.layout.reduce_to(a.shape)
            if b.optional_layout is None:
                mapping[b] = c.layout.reduce_to(b.shape)
            return mapping
        elif a.optional_layout is not None and same_list(a.shape, c.shape):
            return {c: a.layout}
        elif b.optional_layout is not None and same_list(b.shape, c.shape):
            return {c: b.layout}
        else:
            return {}
