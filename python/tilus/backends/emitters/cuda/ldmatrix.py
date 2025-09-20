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
from typing import Mapping

from hidet.ir.dtypes import int32, uint32
from hidet.ir.expr import Expr, cast, deref
from hidet.ir.node import Node
from hidet.ir.primitives.cuda.mma import ldmatrix
from hidet.ir.tools import rewrite

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.ir.instructions import LoadMatrixInst
from tilus.ir.layout import RegisterLayout
from tilus.ir.layout.ops.register_ops import divide
from tilus.ir.utils import vector
from tilus.target import nvgpu_sm75
from tilus.utils import gcd


@register_emitter(LoadMatrixInst, target=nvgpu_sm75)
class LoadMatrixInstEmitter(BaseInstEmitter):
    def emit(self, inst: LoadMatrixInst) -> None:
        tensor = inst.register_output
        layout = tensor.layout
        ldmatrix_layout = inst.config.ldmatrix_layout

        lhs_layout: RegisterLayout = divide(layout, ldmatrix_layout)

        regs_buf = self.get_or_allocate_var(tensor)

        vector_size: int = gcd(lhs_layout.local_size, 4)
        num_vectors: int = lhs_layout.local_size // vector_size

        dtype = inst.output.dtype
        smem_base_addr = self.declare_var("smem_addr", tp=int32, init=inst.smem_addr)

        with self.for_range(num_vectors, attr="u+") as vec_i:
            # load vector_size times of 8x16 bytes of data for each iteration,

            # get the registers
            regs: list[Expr] = []
            for i in range(vector_size):
                regs.append(deref(cast(~regs_buf[(vec_i * vector_size + i) * ldmatrix_layout.local_size], ~uint32)))

            # get the address of each row
            lane_id = self.current_thread % 32
            warp_id = self.current_thread // 32
            lhs_indices = vector(
                lhs_layout.get_global(local_index=vec_i * vector_size + lane_id // 8, spatial_index=warp_id)
            )
            rhs_indices = vector([lane_id % 8, 0])
            rhs_shape = vector(ldmatrix_layout.shape)
            shared_indices = list(lhs_indices * rhs_shape + rhs_indices)

            rewrite_map: Mapping[Node, Node] = {axis: index for axis, index in zip(inst.axes, shared_indices)}
            offset = rewrite(inst.offset, rewrite_map=rewrite_map)
            smem_addr = smem_base_addr + offset * dtype.nbytes

            self.append(ldmatrix(regs=regs, smem_addr=smem_addr, shared_space_addr=True, trans=inst.config.trans))
