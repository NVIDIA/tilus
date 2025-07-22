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
from tilus.ir.instructions import ReduceInst
from tilus.ir.layout import ops
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule


@register_rule(ReduceInst)
class ReduceRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: ReduceInst) -> bool:
        src = inst.register_input
        dst = inst.register_output

        dst_layout = dst.layout

        if not inst.keepdim:
            dst_layout = ops.unsqueeze(dst_layout, dims=[inst.dim])

        return src.layout.reduce_to(dst_layout.shape) == dst_layout
