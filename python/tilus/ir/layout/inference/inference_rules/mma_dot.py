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
from tilus.ir.layout import LayoutOperationError, RegisterLayout, divide
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(DotInst)
class MmaDotRule(LayoutInferenceRule):
    """
    Layout inference rule for MMA dot instructions.
    """

    @staticmethod
    def generate_default_layouts(
        num_warps: int, a: RegisterTensor, b: RegisterTensor, c: RegisterTensor, d: RegisterTensor
    ) -> dict[RegisterTensor, RegisterLayout]:
        from tilus.lang.modules.cuda import cuda

        assert len(a.shape) == len(b.shape) == len(c.shape) == len(d.shape) == 2, "MMA dot requires 2D tensors."

        m = a.shape[0]
        n = b.shape[1]
        k = a.shape[1]

        mma = cuda.resolve_dot_config(operand_dtype=a.dtype, acc_dtype=c.dtype, num_warps=num_warps, m=m, n=n, k=k)
        return {a: mma.la, b: mma.lb, c: mma.lc, d: mma.lc}

    @staticmethod
    def get_atom_config_from_layout_d(
        a: RegisterTensor, b: RegisterTensor, c: RegisterTensor, d: RegisterTensor
    ) -> AtomicMmaConfig:
        from tilus.ir.instructions.cuda.mma_dot import AtomicMmaConfig

        for config in AtomicMmaConfig.all_configs().values():
            if not (a.dtype == b.dtype == config.operand_type and c.dtype == d.dtype == config.acc_type):
                continue

            try:
                divide(d.layout, config.lc)
                if a.optional_layout is not None:
                    divide(a.layout, config.la)
                if b.optional_layout is not None:
                    divide(b.layout, config.lb)
                if c.optional_layout is not None:
                    divide(c.layout, config.lc)
            except LayoutOperationError:
                continue

            return config
        raise ValueError("No suitable MMA configuration found for the given layouts.")

    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: DotInst) -> dict[RegisterTensor, RegisterLayout]:
        a = inst.inputs[0].as_register_tensor()
        b = inst.inputs[1].as_register_tensor()
        c = inst.inputs[2].as_register_tensor()
        d = inst.output.as_register_tensor()

        if all(tensor.optional_layout is None for tensor in (a, b, c, d)):
            num_threads = ctx.num_threads
            if num_threads % 32 != 0:
                raise ValueError("num_threads must be divisible by 32")
            num_warps = num_threads // 32
            return MmaDotRule.generate_default_layouts(num_warps, a, b, c, d)
        elif d.optional_layout is not None:
            # d => a | b | c
            config = MmaDotRule.get_atom_config_from_layout_d(a, b, c, d)
            mapping = {}
            outer_d = divide(d.layout, config.lc)
            m, n = outer_d.shape
            if a.optional_layout is None:
                k = a.shape[1] // (config.k * config.vec_k)
                mapping[a] = outer_d.reduce_to([m, 1]).local(1, k) * config.la
            if b.optional_layout is None:
                k = b.shape[0] // (config.k * config.vec_k)
                mapping[b] = outer_d.reduce_to([1, n]).local(k, 1) * config.lb
            if c.optional_layout is None:
                mapping[c] = d.layout
            return mapping
        else:
            return {}
