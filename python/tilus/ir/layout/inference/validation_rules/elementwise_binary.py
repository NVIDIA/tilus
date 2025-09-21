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
from tilus.ir.instructions import AddInst, DivInst, ElementwiseBinaryInst, Instruction, ModInst, MulInst, SubInst
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule
from tilus.ir.layout.mfunction import ops
from tilus.ir.tensor import RegisterTensor


@register_rule(AddInst)
@register_rule(SubInst)
@register_rule(DivInst)
@register_rule(MulInst)
@register_rule(ModInst)
@register_rule(ElementwiseBinaryInst)
class BinaryRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: Instruction) -> bool:
        assert len(inst.inputs) == 2 and inst.output is not None
        assert all(isinstance(tensor, RegisterTensor) for tensor in inst.inputs + (inst.output,))
        for i in range(2):
            x: RegisterTensor = inst.inputs[i].as_register_tensor()
            y: RegisterTensor = inst.register_output

            fa = ops.identity(y.shape).collapse_by_shape(x.shape) * x.layout.spatial_mfunction()
            fb = y.layout.spatial_mfunction()

            if not fa.cover(fb):
                return False
        return True
