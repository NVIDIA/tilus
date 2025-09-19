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
from typing import Sequence

from hidet.ir import logical_or
from hidet.ir.dtypes import int32, uint32
from hidet.ir.expr import Expr, cast

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.backends.contexts import SharedMemoryAllocationContext, Tcgen05EmitContext
from tilus.extensions.hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from tilus.extensions.hidet.ir.primitives.cuda.tcgen05 import (
    Tcgen05LoadStoreNumKind,
    Tcgen05LoadStorePackKind,
    Tcgen05LoadStoreShapeKind,
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_load,
    tcgen05_relinquish_alloc_permit,
    tcgen05_store,
    tcgen05_wait_load,
    tcgen05_wait_store,
)
from tilus.ir.instructions.cuda.tmem import (
    TMemoryAllocInst,
    TMemoryDeallocInst,
    TMemoryLoadInst,
    TMemoryRelinquishAllocPermitInst,
    TMemorySliceInst,
    TMemoryStoreInst,
    TMemoryViewInst,
    TMemoryWaitInst,
    get_ldst_layout,
)
from tilus.ir.layout.register_layout_ops import divide, left_divide, local, spatial
from tilus.ir.layout.utils import LayoutOperationError
from tilus.ir.tensor import RegisterTensor, TMemoryTensor
from tilus.target import nvgpu_sm100
from tilus.utils import gcd

#    tmem addr: 0xAAAABBBB where AAAA is the lane index and BBBB is the column index
#   lane index: 0x0000 to 0x007F
# column index: 0x0000 to 0x01FF
LANE_STRIDE = 0x00010000
COLUMN_STRIDE = 0x00000001


class Tcgen05AllocDeallocEmitter(BaseInstEmitter):
    def get_num_columns(self, tmem_tensor: TMemoryTensor) -> int:
        assert tmem_tensor.shape[0] == 128
        assert tmem_tensor.shape[1] * tmem_tensor.dtype.nbits % 32 == 0
        num_columns = tmem_tensor.shape[1] * tmem_tensor.dtype.nbits // 32
        assert num_columns % 32 == 0 and 32 <= num_columns <= 512
        return num_columns


@register_emitter(TMemoryAllocInst, target=nvgpu_sm100)
class Tcgen05AllocEmitter(Tcgen05AllocDeallocEmitter):
    def emit(self, inst: TMemoryAllocInst) -> None:
        if self.current_num_threads < 32:
            raise ValueError("tcgen05_alloc requires at least 32 threads in the current thread group")

        tmem_tensor = inst.output.as_tmemory_tensor()
        num_columns = self.get_num_columns(tmem_tensor)

        # set the cta group in the tcgen05 context
        tcgen05_ctx = Tcgen05EmitContext.current()
        tcgen05_ctx.set_cta_group(inst.cta_group)

        # allocate a workspace in shared memory to hold the tensor memory handle
        smem_ctx = SharedMemoryAllocationContext.current()
        smem_ptr = smem_ctx.request_shared_workspace(nbytes=4)

        # call tcgen05_alloc
        with self.if_then(logical_or(self.current_num_threads == 32, self.current_thread // 32 == 0)):
            smem_addr = self.declare_var("smem_addr", tp=uint32, init=cvta_generic_to_shared(smem_ptr))
            self.append(
                tcgen05_alloc(
                    dst=smem_addr,
                    num_columns=uint32(num_columns),
                    cta_group=inst.cta_group,
                )
            )

        # let other warps in the thread group wait until the first warp finishes
        with self.if_then(self.current_num_threads > 32):
            self.sync()

        # load the tensor memory handle from shared memory and store it to the register variable
        tmem_var = self.get_or_allocate_var(tmem_tensor)
        self.assign(tmem_var, cast(smem_ptr, ~int32)[0])
        self.sync()


@register_emitter(TMemoryDeallocInst, target=nvgpu_sm100)
class Tcgen05DeallocEmitter(Tcgen05AllocDeallocEmitter):
    def emit(self, inst: TMemoryDeallocInst) -> None:
        tmem_tensor = inst.inputs[0].as_tmemory_tensor()
        tmem_var = self.get_or_allocate_var(tmem_tensor)
        num_columns = self.get_num_columns(tmem_tensor)
        tcgen05_ctx = Tcgen05EmitContext.current()

        if self.current_num_threads < 32:
            raise ValueError("tcgen05_dealloc requires at least 32 threads in the current thread group")
        with self.if_then(logical_or(self.current_num_threads == 32, self.current_thread // 32 == 0)):
            self.append(
                tcgen05_dealloc(taddr=tmem_var, num_columns=uint32(num_columns), cta_group=tcgen05_ctx.cta_group)
            )


@register_emitter(TMemoryRelinquishAllocPermitInst, target=nvgpu_sm100)
class Tcgen05RelinquishAllocPermitEmitter(BaseInstEmitter):
    def emit(self, inst: TMemoryRelinquishAllocPermitInst) -> None:
        self.append(tcgen05_relinquish_alloc_permit(inst.cta_group))


@register_emitter(TMemorySliceInst, target=nvgpu_sm100)
class TMemorySliceEmitter(BaseInstEmitter):
    def emit(self, inst: TMemorySliceInst) -> None:
        tmem_tensor = inst.inputs[0].as_tmemory_tensor()
        output_tmem_tensor = inst.output.as_tmemory_tensor()
        tmem_addr = self.get_or_allocate_var(tmem_tensor)

        sliced_addr = self.get_or_allocate_var(output_tmem_tensor, name="tmem_slice")
        self.assign(
            sliced_addr,
            tmem_addr + inst.offsets[0] * LANE_STRIDE + inst.offsets[1] * COLUMN_STRIDE * tmem_tensor.dtype.nbits // 32,
        )


@register_emitter(TMemoryViewInst, target=nvgpu_sm100)
class TMemoryViewEmitter(BaseInstEmitter):
    def emit(self, inst: TMemoryViewInst) -> None:
        tmem_tensor = inst.inputs[0].as_tmemory_tensor()
        output_tmem_tensor = inst.output.as_tmemory_tensor()

        if (
            tmem_tensor.dtype.nbits * tmem_tensor.shape[1]
            != output_tmem_tensor.dtype.nbits * output_tmem_tensor.shape[1]
        ):
            raise ValueError("The total number of bits must be the same as the original tensor.")

        tmem_addr = self.get_or_allocate_var(tmem_tensor)
        view_addr = self.get_or_allocate_var(output_tmem_tensor, name="tmem_view")
        self.assign(view_addr, tmem_addr)


@dataclass
class LoadStoreWarpInst:
    taddr: Expr
    regs: list[Expr]
    num_kind: Tcgen05LoadStoreNumKind
    shape_kind: Tcgen05LoadStoreShapeKind
    pack_kind: Tcgen05LoadStorePackKind


class TMemoryLoadStoreBaseEmitter(BaseInstEmitter):
    def slice_tmem_tensor(
        self, tmem_tensor: TMemoryTensor, offsets: Sequence[int], shape: Sequence[int]
    ) -> tuple[TMemoryTensor, Expr]:
        if any(not isinstance(ofs, int) for ofs in offsets):
            raise ValueError("All offsets must be integer constants")
        if len(offsets) != 2:
            raise ValueError("The length of offsets must be 2")
        if len(shape) != 2:
            raise ValueError("The length of shape must be 2")
        tmem_addr = self.get_or_allocate_var(tmem_tensor)
        sliced_tmem_tensor = TMemoryTensor.create(
            dtype=tmem_tensor.dtype, shape=shape, first_lane=tmem_tensor.first_lane + offsets[0]
        )
        sliced_tmem_addr = (
            tmem_addr + offsets[0] * LANE_STRIDE + offsets[1] * COLUMN_STRIDE * tmem_tensor.dtype.nbits // 32
        )
        return sliced_tmem_tensor, sliced_tmem_addr

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
        if self.current_thread_group_begin % 128 != tmem_tensor.first_lane:
            raise ValueError(
                "Lane mismatch: the first lane of the tmem tensor must be the same as the thread group begin"
            )
        if self.current_num_threads != tmem_tensor.shape[0]:
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
            # print("entire_layout: ", entire_layout)
            # print("warp_layout: ", warp_layout)
            # print("warp_repeat: ", warp_repeat)
            # print("atom_layout: ", atom_layout)

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


@register_emitter(TMemoryLoadInst, target=nvgpu_sm100)
class TMemoryLoadEmitter(TMemoryLoadStoreBaseEmitter):
    def emit(self, inst: TMemoryLoadInst) -> None:
        regs_tensor = inst.register_output
        tmem_tensor = inst.inputs[0].as_tmemory_tensor()
        sliced_tmem_tensor, sliced_tmem_addr = self.slice_tmem_tensor(tmem_tensor, inst.offsets, regs_tensor.shape)
        self.emit_tcgen05_instructions(regs_tensor, sliced_tmem_tensor, sliced_tmem_addr)

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


@register_emitter(TMemoryStoreInst, target=nvgpu_sm100)
class TMemoryStoreEmitter(TMemoryLoadStoreBaseEmitter):
    def emit(self, inst: TMemoryStoreInst) -> None:
        regs_tensor = inst.inputs[1].as_register_tensor()
        tmem_tensor = inst.inputs[0].as_tmemory_tensor()

        sliced_tmem_tensor, sliced_tmem_addr = self.slice_tmem_tensor(tmem_tensor, inst.offsets, regs_tensor.shape)
        self.emit_tcgen05_instructions(
            regs_tensor,
            sliced_tmem_tensor,
            sliced_tmem_addr,
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


@register_emitter(TMemoryWaitInst, target=nvgpu_sm100)
class TMemoryWaitEmitter(BaseInstEmitter):
    def emit(self, inst: TMemoryWaitInst) -> None:
        if inst.wait_load:
            self.append(tcgen05_wait_load())
        if inst.wait_store:
            self.append(tcgen05_wait_store())
