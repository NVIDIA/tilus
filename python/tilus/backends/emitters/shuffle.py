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
from hidet.ir.dtypes import boolean, uint8, uint32
from hidet.ir.expr import Expr, Var, bitwise_and, cast, left_shift, logical_and, tensor_pointer_var

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.backends.contexts import SharedMemoryAllocationContext
from tilus.extensions.hidet.ir.primitives.cuda.ldst import load, store
from tilus.ir.instructions.generic import ShuffleBaseInst, ShuffleDownInst, ShuffleUpInst
from tilus.target import nvgpu_any
from tilus.utils import gcd


@register_emitter(ShuffleUpInst, target=nvgpu_any)
@register_emitter(ShuffleDownInst, target=nvgpu_any)
class ShuffleBaseInstEmitter(BaseInstEmitter):
    def emit(self, inst: ShuffleBaseInst) -> None:
        dtype = inst.register_output.dtype
        layout = inst.register_output.layout
        thread_nbytes: int = dtype.nbytes * layout.local_size
        num_groups = max([i // inst.width for i in range(self.num_warps) if inst.mask & (1 << i)]) + 1
        warp_nbytes: int = thread_nbytes * 32

        smem_ctx: SharedMemoryAllocationContext = self.contexts[SharedMemoryAllocationContext]

        smem_nbytes = num_groups * (inst.width - inst.delta) * warp_nbytes
        smem_buf: Var = self.declare(
            v=tensor_pointer_var("shfl_smem", shape=[num_groups, inst.width - inst.delta, warp_nbytes], dtype=uint8),
            init=smem_ctx.request_shared_workspace(nbytes=smem_nbytes),
        )
        warp_id: Expr = self.current_worker // 32
        warp_lane_id = self.current_worker % 32
        group_id = warp_id // inst.width
        shfl_lane_id = warp_id % inst.width

        cond_in_mask = cast(bitwise_and(inst.mask, left_shift(1, warp_id)), boolean)
        if isinstance(inst, ShuffleDownInst):
            cond_is_sender = shfl_lane_id >= inst.delta
            cond_is_receiver = shfl_lane_id < inst.width - inst.delta
            sender_shfl_lane = shfl_lane_id - inst.delta
            receiver_shfl_lane = shfl_lane_id
        else:
            cond_is_sender = shfl_lane_id < inst.width - inst.delta
            cond_is_receiver = shfl_lane_id >= inst.delta
            sender_shfl_lane = shfl_lane_id
            receiver_shfl_lane = shfl_lane_id - inst.delta
        # store the data from the register to shared memory
        src_var: Var = self.tensor2var[inst.inputs[0]]
        with self.if_then(logical_and(cond_in_mask, cond_is_sender)):
            if thread_nbytes % 4 == 0:
                vec: int = gcd(thread_nbytes // 4, 4)
                vec_nbytes: int = vec * 4

                with self.for_range(thread_nbytes // vec_nbytes) as i:
                    self.append(
                        store(
                            dtype=uint32,
                            addr=~smem_buf[
                                group_id, sender_shfl_lane, i * (32 * vec_nbytes) + warp_lane_id * vec_nbytes
                            ],
                            src_addrs=[cast(src_var, ~uint8) + j * 4 for j in range(vec)],
                        )
                    )
            else:
                raise NotImplementedError()
        self.sync()

        # load the data from shared memory to register
        dst_var: Var = self.get_or_allocate_var(inst.register_output, "shuffled")
        with self.if_then(logical_and(cond_in_mask, cond_is_receiver)):
            if thread_nbytes % 4 == 0:
                vec = gcd(thread_nbytes // 4, 4)
                vec_nbytes = vec * 4

                with self.for_range(thread_nbytes // vec_nbytes) as i:
                    self.append(
                        load(
                            dtype=uint32,
                            addr=~smem_buf[
                                group_id, receiver_shfl_lane, i * (32 * vec_nbytes) + warp_lane_id * vec_nbytes
                            ],
                            dst_addrs=[cast(dst_var, ~uint8) + j * 4 for j in range(vec)],
                        )
                    )
            else:
                raise NotImplementedError()
        self.sync()
