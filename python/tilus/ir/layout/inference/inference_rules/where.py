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

from tilus import RegisterLayout
from tilus.ir.instructions import WhereInst
from tilus.ir.layout import ops
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(WhereInst)
class WhereRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: WhereInst) -> dict[RegisterTensor, RegisterLayout]:
        cond = inst.inputs[0].as_register_tensor()
        x = inst.inputs[1].as_register_tensor()
        y = inst.inputs[2].as_register_tensor()
        out = inst.register_output

        if out.optional_layout is not None:
            ret = {}
            for operand in (cond, x, y):
                if operand.optional_layout is None:
                    ret[operand] = ops.reduce_to(out.layout, shape=operand.shape)
            return ret
        else:
            for operand in (cond, x, y):
                if operand.optional_layout is not None and same_list(operand.shape, out.shape):
                    return {out: operand.layout}
        return {}
