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
from tilus.ir.instructions import DotInst
from tilus.ir.instructions.cuda.mma_dot import AtomicMmaConfig
from tilus.ir.layout import LayoutOperationError
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule
from tilus.ir.layout.ops import divide


@register_rule(DotInst)
class MmaDotRule(LayoutValidationRule):
    """
    Layout inference rule for MMA dot instructions.
    """

    @staticmethod
    def validate(inst: DotInst) -> bool:
        from tilus.ir.layout.mfunction.ops import identity

        a = inst.inputs[0].as_register_tensor()
        b = inst.inputs[1].as_register_tensor()
        c = inst.inputs[2].as_register_tensor()
        d = inst.output.as_register_tensor()
        for config in AtomicMmaConfig.all_configs().values():
            if not (a.dtype == b.dtype == config.operand_type and c.dtype == d.dtype == config.acc_type):
                continue

            try:
                outer_a = divide(a.layout, config.la)
                outer_b = divide(b.layout, config.lb)
                outer_c = divide(c.layout, config.lc)
                outer_d = divide(d.layout, config.lc)
            except LayoutOperationError:
                continue

            m, n, k = outer_d.shape[0], outer_d.shape[1], outer_a.shape[1]

            mf_g = identity([m, k, n])
            mf_a = mf_g.collapse(dims=[2]) * outer_a.spatial_mfunction()
            mf_b = mf_g.collapse(dims=[0]) * outer_b.spatial_mfunction()
            mf_c = mf_g.collapse(dims=[1]) * outer_c.spatial_mfunction()
            mf_d = mf_g.collapse(dims=[1]) * outer_d.spatial_mfunction()

            if any(not mf_operand.cover(mf_d) for mf_operand in (mf_a, mf_b, mf_c)):
                continue

            return True

        return False
