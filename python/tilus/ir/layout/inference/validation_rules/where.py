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
from tilus.ir.instructions import WhereInst
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule
from tilus.ir.layout.mfunction import ops
from tilus.ir.tensor import RegisterTensor


@register_rule(WhereInst)
class WhereRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: WhereInst) -> bool:
        cond: RegisterTensor = inst.inputs[0].as_register_tensor()
        x: RegisterTensor = inst.inputs[1].as_register_tensor()
        y: RegisterTensor = inst.inputs[2].as_register_tensor()

        for operand in [cond, x, y]:
            out: RegisterTensor = inst.register_output

            fa = ops.identity(out.shape).collapse_by_shape(operand.shape) * operand.layout.spatial_mfunction()
            fb = out.layout.spatial_mfunction()

            if not fa.cover(fb):
                return False
        return True
