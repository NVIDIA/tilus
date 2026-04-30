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
"""Emitter for Philox-4x32 random number generation instruction."""

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.hidet.ir.dtypes import uint32, uint64
from tilus.hidet.ir.expr import BitwiseAnd
from tilus.hidet.ir.primitives.cuda.random import philox4x32
from tilus.ir.instructions import Philox4x32Inst
from tilus.ir.tensor import RegisterTensor


@register_emitter(Philox4x32Inst)
class Philox4x32InstEmitter(BaseInstEmitter):
    def emit(self, inst: Philox4x32Inst) -> None:
        offset_tensor: RegisterTensor = inst.inputs[0].as_register_tensor()
        output_tensor: RegisterTensor = inst.register_output
        offset_buf = self.tensor2var[offset_tensor]
        output_buf = self.get_or_allocate_var(output_tensor)

        seed_expr = inst.seed
        n = offset_tensor.local_size

        # Split seed (uint64) into two uint32 halves: k0 = low, k1 = high
        seed_lo = self.declare_var("seed_lo", tp=uint32, init=BitwiseAnd(seed_expr, uint64(0xFFFFFFFF)))
        seed_hi = self.declare_var("seed_hi", tp=uint32, init=BitwiseAnd(seed_expr >> uint64(32), uint64(0xFFFFFFFF)))

        # Since output layout = local(4, 1, ...) * offset_layout, the local buffer is laid out as:
        #   output_buf[4 * n]  where  output_buf[j * n + i] <=> component j of offset element i
        with self.for_range(extent=n) as i:
            self.append(
                philox4x32(
                    seed_lo,
                    seed_hi,
                    offset=offset_buf[i],
                    out0=~output_buf[0 * n + i],
                    out1=~output_buf[1 * n + i],
                    out2=~output_buf[2 * n + i],
                    out3=~output_buf[3 * n + i],
                )
            )
