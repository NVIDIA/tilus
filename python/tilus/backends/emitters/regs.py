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

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.ir import RegisterTensor
from tilus.ir.instructions import SliceAssignInst, SliceRegisterInst
from tilus.ir.layout import RegisterLayout


@register_emitter(SliceRegisterInst)
class SliceRegisterInstEmitter(BaseInstEmitter):
    def emit(self, inst: SliceRegisterInst) -> None:  # type: ignore
        dst_tensor: RegisterTensor = inst.register_output
        dst_layout: RegisterLayout = dst_tensor.layout
        src_tensor: RegisterTensor = inst.register_input
        src_layout: RegisterLayout = src_tensor.layout

        const_reg_ctx = self.contexts.const_reg_ctx

        if const_reg_ctx.is_tracked(src_tensor):
            self._emit_const_reg(inst, dst_tensor, dst_layout, src_tensor, src_layout, const_reg_ctx)
        else:
            self._emit_default(inst, dst_tensor, dst_layout, src_tensor, src_layout)

    def _emit_default(self, inst, dst_tensor, dst_layout, src_tensor, src_layout):
        """Default emission: materialize as array indexing."""
        dst_var = self.get_or_allocate_var(tensor=dst_tensor, name="slice_regs")
        src_var = self.tensor2var[src_tensor]

        with self.for_range(extent=dst_layout.local_size) as dst_local:
            dst_indices = src_layout.get_global(spatial_index=self.current_thread, local_index=dst_local)
            src_indices = list(inst.offsets)
            dims = range(len(src_layout.shape)) if inst.dims is None else inst.dims
            for dim in dims:
                src_indices[dim] = src_indices[dim] + dst_indices[dim]
            src_local = src_layout.get_local(global_indices=src_indices)
            self.buffer_store(buf=dst_var, indices=[dst_local], value=src_var[src_local])

    def _emit_const_reg(self, inst, dst_tensor, dst_layout, src_tensor, src_layout, const_reg_ctx):
        """Optimized emission: compute values from logical index expressions, bypassing array indexing."""
        from tilus.hidet.ir.dtypes import int32
        from tilus.hidet.ir.expr import Var

        dst_var = self.get_or_allocate_var(tensor=dst_tensor, name="slice_regs")

        # For each destination element, compute the logical source indices from the offsets
        with self.for_range(extent=dst_layout.local_size) as dst_local:
            # Get the logical indices of the destination element in the source tensor's index space
            dst_global = dst_layout.get_global(spatial_index=self.current_thread, local_index=dst_local)
            src_logical_indices = list(inst.offsets)
            dims = range(len(src_layout.shape)) if inst.dims is None else inst.dims
            for dim in dims:
                src_logical_indices[dim] = src_logical_indices[dim] + dst_global[dim]

            value = const_reg_ctx.get_value(src_tensor, src_logical_indices)
            self.buffer_store(buf=dst_var, indices=[dst_local], value=value)

        # Propagate CTA-invariant tracking to the output tensor
        dst_axes = [Var(f"i{d}", type=int32) for d in range(len(dst_tensor.shape))]
        dst_global = dst_layout.get_global(
            spatial_index=self.current_thread, local_index=dst_axes[0] if dst_axes else int32(0)
        )
        src_logical_indices = list(inst.offsets)
        dims = range(len(src_layout.shape)) if inst.dims is None else inst.dims
        for dim in dims:
            src_logical_indices[dim] = src_logical_indices[dim] + dst_global[dim]
        dst_expr = const_reg_ctx.get_value(src_tensor, src_logical_indices)
        const_reg_ctx.register(dst_tensor, axes=dst_axes, expr=dst_expr)


@register_emitter(SliceAssignInst)
class SliceAssignInstEmitter(BaseInstEmitter):
    def emit(self, inst: SliceAssignInst) -> None:  # type: ignore
        src_tensor: RegisterTensor = inst.inputs[1].as_register_tensor()
        src_layout: RegisterLayout = src_tensor.layout
        dst_tensor: RegisterTensor = inst.inputs[0].as_register_tensor()
        dst_layout: RegisterLayout = dst_tensor.layout

        dst_var = self.tensor2var[dst_tensor]
        src_var = self.tensor2var[src_tensor]

        with self.for_range(extent=src_layout.local_size) as src_local:
            src_indices = src_layout.get_global(spatial_index=self.current_thread, local_index=src_local)
            dst_indices = list(inst.offsets)
            dims = range(len(src_layout.shape)) if inst.dims is None else inst.dims
            for dim in dims:
                dst_indices[dim] = dst_indices[dim] + src_indices[dim]
            dst_local = dst_layout.get_local(global_indices=dst_indices)
            self.buffer_store(buf=dst_var, indices=[dst_local], value=src_var[src_local])

        self.tensor2var[dst_tensor] = dst_var
