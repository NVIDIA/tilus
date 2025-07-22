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
from __future__ import annotations

from hidet.ir.dtypes import uint32
from hidet.ir.expr import Expr, cast, if_then_else, var

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.extensions.hidet.ir.primitives.cuda.mma import mma_sync_v2
from tilus.ir.instructions.cuda import DotInst
from tilus.ir.instructions.cuda.mma_dot import AtomicMmaConfig
from tilus.ir.layout import LayoutOperationError, RegisterLayout
from tilus.ir.tensor import RegisterTensor
from tilus.target import nvgpu_sm70


@register_emitter(DotInst, target=nvgpu_sm70)
class DotInstEmitter(BaseInstEmitter):
    @staticmethod
    def resolve_mma_config(
        a: RegisterTensor, b: RegisterTensor, c: RegisterTensor, d: RegisterTensor
    ) -> tuple[AtomicMmaConfig, tuple[RegisterLayout, ...]]:
        assert a.dtype == b.dtype and c.dtype == d.dtype
        for config in AtomicMmaConfig.all_configs().values():
            if a.dtype != config.operand_type or c.dtype != config.acc_type:
                # dtype mismatch, skip
                continue
            try:
                outers = [
                    p / q
                    for p, q in zip(
                        [a.layout, b.layout, c.layout, d.layout], [config.la, config.lb, config.lc, config.lc]
                    )
                ]
            except LayoutOperationError:
                # layout not divisible, skip
                continue

            return config, tuple(outers)

        msg = (
            "Can not resolve the mma config for\n"
            + " a: {}{} {}\n".format(a.dtype.name, list(a.shape), a.layout)
            + " b: {}{} {}\n".format(b.dtype.name, list(b.shape), b.layout)
            + " c: {}{} {}\n".format(c.dtype.name, list(c.shape), c.layout)
            + " d: {}{} {}\n".format(d.dtype.name, list(d.shape), d.layout)
        )
        raise RuntimeError(msg)

    def emit(self, inst: DotInst) -> None:  # type: ignore
        a_tensor = inst.inputs[0].as_register_tensor()
        b_tensor = inst.inputs[1].as_register_tensor()
        c_tensor = inst.inputs[2].as_register_tensor()
        d_tensor = inst.register_output
        config, outers = self.resolve_mma_config(a_tensor, b_tensor, c_tensor, d_tensor)
        a_buf = self.tensor2var[a_tensor]
        b_buf = self.tensor2var[b_tensor]
        c_buf = self.tensor2var[c_tensor]
        d_buf = self.get_or_allocate_var(tensor=d_tensor, name="d")
        outer_a, outer_b, outer_c, outer_d = outers
        k_size = outer_a.shape[1]

        warp_id: Expr = self.current_worker // 32
        with self.for_range(outer_c.local_size) as c_outer_local:
            c_outer_indices = outer_c.get_global(local_index=c_outer_local, spatial_index=warp_id)
            d_outer_indices = c_outer_indices
            c_local = c_outer_local * config.lc.local_size
            d_local = outer_d.get_local(d_outer_indices) * config.lc.local_size
            with self.for_range(k_size) as k:
                i, j = c_outer_indices
                a_outer_indices = (i, k)
                b_outer_indices = (k, j)
                a_local = outer_a.get_local(a_outer_indices) * config.la.local_size
                b_local = outer_b.get_local(b_outer_indices) * config.lb.local_size

                with self.for_range(config.vec_k) as k_inner:
                    a_regs = self.declare(var("a_regs", ~uint32), init=cast(~a_buf[a_local], ~uint32))
                    b_regs = self.declare(var("b_regs", ~uint32), init=cast(~b_buf[b_local], ~uint32))
                    if c_tensor is d_tensor:
                        c_regs = self.declare(var("c_regs", ~uint32), init=cast(~c_buf[c_local], ~uint32))
                    else:
                        c_regs = self.declare(
                            var("c_regs", ~uint32),
                            init=cast(if_then_else(k == 0, ~c_buf[c_local], ~d_buf[d_local]), ~uint32),
                        )
                    d_regs = self.declare(var("d_regs", ~uint32), init=cast(~d_buf[d_local], ~uint32))

                    self.append(
                        mma_sync_v2(
                            config=config.hidet_mma_config(),
                            d_reg_p=[d_regs + i for i in range(config.hidet_mma_config().c_regs)],
                            a_reg_p=[
                                a_regs + i * config.vec_k + k_inner for i in range(config.hidet_mma_config().a_regs)
                            ],
                            b_reg_p=[
                                b_regs + i * config.vec_k + k_inner for i in range(config.hidet_mma_config().b_regs)
                            ],
                            c_reg_p=[c_regs + i for i in range(config.hidet_mma_config().c_regs)],
                        )
                    )
