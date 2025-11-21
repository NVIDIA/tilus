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

from dataclasses import dataclass

from hidet.ir.dtypes import int32, uint32
from hidet.ir.expr import Expr, cast

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import (
    COLUMN_STRIDE,
    LANE_STRIDE,
    Tcgen05LoadStoreNumKind,
    Tcgen05LoadStorePackKind,
    Tcgen05LoadStoreShapeKind,
    tcgen05_load,
    tcgen05_store,
    tcgen05_wait_load,
    tcgen05_wait_store,
)
from tilus.ir.instructions.cuda.tcgen05 import (
    Tcgen05LoadInst,
    Tcgen05StoreInst,
    Tcgen05WaitInst,
)
from tilus.ir.layout.cuda.tcgen05.ldst import get_ldst_layout
from tilus.ir.layout.ops.register_ops import divide, left_divide, local, spatial
from tilus.ir.layout.ops.utils import LayoutOperationError
from tilus.ir.tensor import RegisterTensor, TMemoryTensor
from tilus.target import nvgpu_sm100
from tilus.utils import gcd


@dataclass
class LoadStoreWarpInst:
    taddr: Expr
    regs: list[Expr]
    num_kind: Tcgen05LoadStoreNumKind
    shape_kind: Tcgen05LoadStoreShapeKind
    pack_kind: Tcgen05LoadStorePackKind


class TMemoryLoadStoreBaseEmitter(BaseInstEmitter):
    def emit_tcgen05_inst(self, inst: LoadStoreWarpInst) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def emit_tcgen05_instructions(
        self,
        regs_tensor: RegisterTensor,
        tmem_tensor: TMemoryTensor,
        tmem_base_addr: Expr,
    ) -> None:
        if self.current_num_threads % 32 != 0:
            raise ValueError("The number of threads in the current thread group must be divisible by 32")
        if self.current_thread_group_begin % 128 != tmem_tensor.layout.lane_offset:
            raise ValueError(
                "Lane mismatch: the first lane of the tmem tensor must be the same as the thread group begin"
            )
        if self.current_num_threads != tmem_tensor.shape[-2]:
            raise ValueError(
                "The number of threads in the current thread group must be the same as the number of lanes in the tmem tensor"
            )
        if regs_tensor.dtype.nbits > 32 or 32 % regs_tensor.dtype.nbits != 0:
            raise NotImplementedError("Only 8-bit, 16-bit, and 32-bit data types are supported")
        num_warps = self.current_num_threads // 32
        entire_layout = regs_tensor.layout

        num_elements_per_register = 32 // regs_tensor.dtype.nbits

        # get the entire layout with 32-bit register as the element type
        entire_layout = divide(entire_layout, rhs=local(1, num_elements_per_register))

        # get the layout for each warp
        warp_layout = left_divide(entire_layout, lhs_divisor=spatial(num_warps, 1))

        # try different atom layouts supported by tcgen05
        for shape_kind in [
            Tcgen05LoadStoreShapeKind.R32x32B,
            Tcgen05LoadStoreShapeKind.R16x64B,
            Tcgen05LoadStoreShapeKind.R16x128B,
            Tcgen05LoadStoreShapeKind.R16x256B,
        ]:
            atom_layout = get_ldst_layout(shape_kind)
            try:
                warp_repeat = divide(warp_layout, atom_layout)
                warp_repeat_m, warp_repeat_n = warp_repeat.shape
                assert warp_repeat == local(warp_repeat_m, warp_repeat_n)
            except LayoutOperationError:
                continue

            # now we have
            # entire_layout = spatial(num_warps, 1) * warp_layout
            # warp_layout = warp_repeat * atom_layout
            # each atom_layout corresponds to a warp-level tcgen05 load/store instruction

            # get the .num for the instruction
            num: int = gcd(warp_repeat_n, 128 // shape_kind.regs_per_thread())

            regs_buf = self.declare_var(
                "regs_buf", tp=~uint32, init=cast(~self.get_or_allocate_var(regs_tensor)[0], ~uint32)
            )
            warp_tmem_base_addr = self.declare_var(
                "warp_tmem_base_addr", tp=~int32, init=tmem_base_addr + self.current_thread // 32 * 32 * LANE_STRIDE
            )

            with self.for_range(warp_repeat_m, attr="u") as warp_repeat_i:
                with self.for_range(warp_repeat_n // num, attr="u") as warp_repeat_vec_j:
                    # get the tmem address for each instruction
                    atom_addr = (
                        warp_tmem_base_addr + warp_repeat_i * LANE_STRIDE + warp_repeat_vec_j * num * COLUMN_STRIDE
                    )

                    # get the registers for the instruction
                    regs = []
                    num_regs = atom_layout.local_size
                    for wj in range(num):
                        warp_repeat_j = warp_repeat_vec_j * num + wj
                        for i in range(num_regs):
                            regs.append(
                                ~regs_buf[(warp_repeat_i * warp_repeat_n + warp_repeat_j) * atom_layout.local_size + i]
                            )

                    self.emit_tcgen05_inst(
                        LoadStoreWarpInst(
                            taddr=atom_addr,
                            regs=regs,
                            num_kind=Tcgen05LoadStoreNumKind.from_int(num),
                            shape_kind=shape_kind,
                            pack_kind=Tcgen05LoadStorePackKind.NONE,
                        )
                    )
            return None

        raise RuntimeError("No valid tcgen05 load/store instruction is found")


@register_emitter(Tcgen05LoadInst, target=nvgpu_sm100)
class TMemoryLoadEmitter(TMemoryLoadStoreBaseEmitter):
    def emit(self, inst: Tcgen05LoadInst) -> None:
        regs_tensor = inst.register_output
        tmem_tensor = inst.inputs[0].as_tmemory_tensor()
        self.emit_tcgen05_instructions(regs_tensor, tmem_tensor, self.tensor2var[tmem_tensor])

    def emit_tcgen05_inst(self, inst: LoadStoreWarpInst) -> None:
        self.append(
            tcgen05_load(
                taddr=inst.taddr,
                regs=inst.regs,
                shape=inst.shape_kind,
                num=inst.num_kind,
                pack=inst.pack_kind,
            )
        )


@register_emitter(Tcgen05StoreInst, target=nvgpu_sm100)
class TMemoryStoreEmitter(TMemoryLoadStoreBaseEmitter):
    def emit(self, inst: Tcgen05StoreInst) -> None:
        regs_tensor = inst.inputs[1].as_register_tensor()
        tmem_tensor = inst.inputs[0].as_tmemory_tensor()

        self.emit_tcgen05_instructions(
            regs_tensor,
            tmem_tensor,
            self.tensor2var[tmem_tensor],
        )

    def emit_tcgen05_inst(self, inst: LoadStoreWarpInst) -> None:
        self.append(
            tcgen05_store(
                taddr=inst.taddr,
                regs=inst.regs,
                shape=inst.shape_kind,
                num=inst.num_kind,
                pack=inst.pack_kind,
            )
        )


@register_emitter(Tcgen05WaitInst, target=nvgpu_sm100)
class TMemoryWaitEmitter(BaseInstEmitter):
    def emit(self, inst: Tcgen05WaitInst) -> None:
        if inst.wait_load:
            self.append(tcgen05_wait_load())
        if inst.wait_store:
            self.append(tcgen05_wait_store())
