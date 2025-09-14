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
from typing import Sequence, no_type_check

from hidet.ir.expr import Expr
from hidet.ir.primitives.cuda.funcs import call_cuda
from hidet.ir.primitives.cuda.mma import MmaConfig, mma_configs
from hidet.ir.primitives.func import register_primitive_function
from hidet.ir.stmt import asm
from hidet.utils import initialize

from tilus.extensions.hidet.ir.expr import deref


@initialize()
def register_mma_instructions():
    for config in mma_configs.values():
        inst_name = config.inst_name()

        a_regs, b_regs, c_regs, d_regs = config.a_regs, config.b_regs, config.c_regs, config.c_regs
        template_sub_strings = [
            inst_name,
            "{{{}}},".format(", ".join([f"%{i}" for i in range(d_regs)])),
            "{{{}}},".format(", ".join([f"%{i}" for i in range(d_regs, d_regs + a_regs)])),
            "{{{}}},".format(", ".join([f"%{i}" for i in range(d_regs + a_regs, d_regs + a_regs + b_regs)])),
            "{{{}}};".format(
                ", ".join([f"%{i}" for i in range(d_regs + a_regs + b_regs, d_regs + a_regs + b_regs + c_regs)])
            ),
        ]
        template_string = " ".join(template_sub_strings)

        # v1
        func_name = "cuda_" + inst_name.replace(".", "_")

        # v2
        from hidet.lang import attrs, meta, script
        from hidet.lang.types import uint32, void_p

        a_reg_p_type = meta.types([void_p for _ in range(a_regs)])
        b_reg_p_type = meta.types([void_p for _ in range(b_regs)])
        c_reg_p_type = meta.types([void_p for _ in range(c_regs)])
        d_reg_p_type = meta.types([void_p for _ in range(d_regs)])

        @no_type_check
        @script
        def mma_sync_v2_primitive(
            d_reg_p: d_reg_p_type, a_reg_p: a_reg_p_type, b_reg_p: b_reg_p_type, c_reg_p: c_reg_p_type
        ):
            attrs.func_name = func_name + "_v2"
            attrs.func_kind = "cuda_internal"

            asm(
                template_string,
                outputs=[deref(d_reg_p[i], uint32) for i in range(d_regs)],
                inputs=[deref(a_reg_p[i], uint32) for i in range(a_regs)]
                + [deref(b_reg_p[i], uint32) for i in range(b_regs)]
                + [deref(c_reg_p[i], uint32) for i in range(c_regs)],
                is_volatile=True,
            )

        register_primitive_function(mma_sync_v2_primitive.name, mma_sync_v2_primitive)


def mma_sync_v2(
    config: MmaConfig,
    d_reg_p: Sequence[Expr],
    a_reg_p: Sequence[Expr],
    b_reg_p: Sequence[Expr],
    c_reg_p: Sequence[Expr],
) -> Expr:
    name = config.inst_name().replace(".", "_") + "_v2"
    return call_cuda(func_name=name, args=[*d_reg_p, *a_reg_p, *b_reg_p, *c_reg_p])
