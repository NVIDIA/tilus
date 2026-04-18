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
"""
Unified emitter for LoadSharedInst and StoreSharedInst.

For LoadSharedInst:
- First attempts to emit ldmatrix (checks compatibility via LoadMatrixConfig)
- Falls back to generic element-wise loads with vectorization

For StoreSharedInst:
- Emits generic element-wise stores with vectorization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional

if TYPE_CHECKING:
    from tilus.ir.func import Analysis

from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.hidet.ir.dtypes import boolean, int32, uint8, uint16, uint32
from tilus.hidet.ir.dtypes.vector import uint32x2, uint32x4
from tilus.hidet.ir.expr import Expr, Var, as_expr, cast, deref, if_then_else, index_vars
from tilus.hidet.ir.node import Node
from tilus.hidet.ir.primitives.cuda.mma import ldmatrix, stmatrix
from tilus.hidet.ir.tools import collect
from tilus.hidet.ir.tools.rewriter import rewrite
from tilus.hidet.ir.type import DataType
from tilus.ir.analyzers.grid_analyzer import TensorInfo, analyze_grid
from tilus.ir.instructions import LoadSharedInst, StoreSharedInst
from tilus.ir.layout import RegisterLayout
from tilus.ir.layout.cuda.ldmatrix import LoadMatrixConfig, StoreMatrixConfig
from tilus.ir.layout.ops import divide
from tilus.ir.layout.ops.utils import LayoutOperationError
from tilus.ir.tensor import RegisterTensor, SharedTensor
from tilus.ir.utils import vector
from tilus.target import nvgpu_sm75, nvgpu_sm90
from tilus.utils import gcd


def _get_load_matrix_config(dtype: DataType, register_layout: RegisterLayout) -> Optional[LoadMatrixConfig]:
    """Check if the register layout is compatible with ldmatrix."""
    if len(register_layout.shape) != 2:
        return None
    for config in LoadMatrixConfig.all():
        if dtype.nbytes != config.nbytes:
            continue
        try:
            divide(register_layout, config.ldmatrix_layout)
        except LayoutOperationError:
            continue
        return config
    return None


def _get_store_matrix_config(dtype: DataType, register_layout: RegisterLayout) -> Optional[StoreMatrixConfig]:
    """Check if the register layout is compatible with stmatrix."""
    if len(register_layout.shape) != 2:
        return None
    for config in StoreMatrixConfig.all():
        if dtype.nbytes != config.nbytes:
            continue
        try:
            divide(register_layout, config.stmatrix_layout)
        except LayoutOperationError:
            continue
        return config
    return None


def _check_shared_alignment_and_contiguity(
    shared_tensor: SharedTensor,
    register_shape: tuple[int, ...],
    analysis: Optional["Analysis"],
    config_nbytes: int,
    unit_shape_last: int,
) -> bool:
    """Check alignment and contiguity of shared tensor for ldmatrix/stmatrix compatibility."""
    axes: list[Var] = index_vars(num_vars=len(shared_tensor.shape))
    offset: Expr = shared_tensor.layout(*axes)
    offset_used_vars = collect(offset, [Var], stop_when_found=True)
    var2info = {}
    shape = register_shape
    for v in offset_used_vars:
        if v in axes:
            continue
        if analysis is None:
            continue
        if (
            v in analysis.lower_bound
            and v in analysis.upper_bound
            and analysis.lower_bound[v] == analysis.upper_bound[v]
        ):
            var2info[v] = TensorInfo.from_constant(shape=shape, value=analysis.lower_bound[v])
        elif v in analysis.divisibility:
            var2info[v] = TensorInfo.from_divisibility(shape=shape, divisibility=analysis.divisibility[v])
    tensor_info: TensorInfo = analyze_grid(shape=shape, axes=axes, expr=offset, var2info=var2info)

    if tensor_info.infos[-1].divisibility * config_nbytes % 16 != 0:
        return False
    if tensor_info.infos[-1].continuity % unit_shape_last != 0:
        return False
    return True


def _analyze_vectorization_for_shared(
    register_tensor: RegisterTensor,
    shared_tensor: SharedTensor,
    analysis: Optional["Analysis"],
    is_load: bool,
) -> Optional[tuple[int, int]]:
    """
    Analyze vectorization for generic shared ld/st.

    Returns (vectorize_dimension, vector_bytes) or None if vectorization is not possible.
    """
    dtype = register_tensor.dtype
    layout = register_tensor.layout
    shape = layout.shape
    rank = len(shape)

    # Compute axes, offset, mask from shared layout (merged from lower_load_store.py)
    axes = tuple(index_vars(num_vars=rank))
    offset_expr = shared_tensor.layout(*axes)
    mask_expr = as_expr(boolean.true)

    # Analyze offset and mask value information
    offset_info = analyze_grid(shape=shape, axes=axes, expr=offset_expr, analysis=analysis)
    mask_info = analyze_grid(shape=shape, axes=axes, expr=mask_expr, analysis=analysis)

    # Analyze register layout
    layout_axes = index_vars(len(layout.shape))
    layout_expr = layout.get_local(global_indices=layout_axes)
    layout_info: TensorInfo = analyze_grid(
        shape=layout.shape, axes=layout_axes, analysis=analysis, expr=as_expr(layout_expr)
    )

    # Enumerate dimensions for vectorization
    for i in range(len(shape)):
        max_vector_elements = gcd(
            offset_info[i].divisibility,
            offset_info[i].continuity,
            layout_info[i].continuity,
            mask_info[i].constancy,
            layout.local_size,
        )
        if max_vector_elements > 1 and max_vector_elements * dtype.nbits % 8 == 0:
            vector_bytes = max_vector_elements * dtype.nbits // 8
            return (i, vector_bytes)

    return None


def _emit_generic_load_shared(emitter: BaseInstEmitter, inst: LoadSharedInst) -> None:
    """Emit generic element-wise loads with vectorization (merged from ldst.py)."""
    register_tensor = inst.register_output
    shared_tensor = inst.shared_input
    dtype = register_tensor.dtype
    layout = register_tensor.layout

    regs_buf = emitter.get_or_allocate_var(register_tensor)
    smem_buf = emitter.get_or_allocate_var(shared_tensor)

    # Compute axes, offset, mask from shared layout
    rank = len(register_tensor.shape)
    axes = tuple(index_vars(num_vars=rank))
    offset_expr = shared_tensor.layout(*axes)
    mask_expr = as_expr(boolean.true)

    vectorization = _analyze_vectorization_for_shared(
        register_tensor=register_tensor,
        shared_tensor=shared_tensor,
        analysis=emitter.analysis,
        is_load=True,
    )

    if vectorization:
        vectorize_dimension, vector_bytes = vectorization
        total_nbytes = layout.local_size * dtype.nbits // 8
        with emitter.for_range(extent=total_nbytes // vector_bytes) as vec_i:
            start_i = vec_i * vector_bytes * 8 // dtype.nbits
            global_indices = layout.get_global(local_index=start_i, spatial_index=emitter.current_thread)
            rewrite_map = {axis: as_expr(global_index) for axis, global_index in zip(axes, global_indices)}
            offset = rewrite(offset_expr, rewrite_map=rewrite_map)
            mask = rewrite(mask_expr, rewrite_map=rewrite_map)

            unit_bytes: int = gcd(vector_bytes, 16)
            unit_dtype: DataType = {1: uint8, 2: uint16, 4: uint32, 8: uint32x2, 16: uint32x4}[unit_bytes]
            num_units: int = vector_bytes // unit_bytes

            reg_ptr = emitter.declare_var("reg_ptr", ~unit_dtype, init=cast(~regs_buf[start_i], ~unit_dtype))
            mem_ptr = emitter.declare_var("mem_ptr", ~unit_dtype, init=cast(~smem_buf[offset], ~unit_dtype))
            dst_ptr, src_ptr = reg_ptr, mem_ptr
            with emitter.if_then(mask):
                with emitter.for_range(extent=num_units) as i:
                    emitter.buffer_store(buf=dst_ptr, indices=[i], value=src_ptr[i])
            with emitter.otherwise():
                with emitter.for_range(extent=num_units) as i:
                    emitter.buffer_store(buf=dst_ptr, indices=[i], value=unit_dtype.zero)
    else:
        with emitter.for_range(extent=register_tensor.local_size) as i:
            global_indices = layout.get_global(local_index=i, spatial_index=emitter.current_thread)
            rewrite_map = {axis: as_expr(global_index) for axis, global_index in zip(axes, global_indices)}
            offset = rewrite(offset_expr, rewrite_map=rewrite_map)
            mask = rewrite(mask_expr, rewrite_map=rewrite_map)
            emitter.buffer_store(buf=regs_buf, indices=[i], value=if_then_else(mask, smem_buf[offset], dtype.zero))


def _emit_generic_store_shared(emitter: BaseInstEmitter, inst: StoreSharedInst) -> None:
    """Emit generic element-wise stores with vectorization (merged from ldst.py)."""
    shared_tensor = inst.inputs[0].as_shared_tensor()
    register_tensor = inst.inputs[1].as_register_tensor()
    dtype = register_tensor.dtype
    layout = register_tensor.layout

    regs_buf = emitter.get_or_allocate_var(register_tensor)
    smem_buf = emitter.get_or_allocate_var(shared_tensor)

    # Compute axes, offset, mask from shared layout
    rank = len(register_tensor.shape)
    axes = tuple(index_vars(num_vars=rank))
    offset_expr = shared_tensor.layout(*axes)
    mask_expr = as_expr(boolean.true)

    vectorization = _analyze_vectorization_for_shared(
        register_tensor=register_tensor,
        shared_tensor=shared_tensor,
        analysis=emitter.analysis,
        is_load=False,
    )

    if vectorization:
        vectorize_dimension, vector_bytes = vectorization
        total_nbytes = layout.local_size * dtype.nbits // 8
        with emitter.for_range(extent=total_nbytes // vector_bytes) as vec_i:
            start_i = vec_i * vector_bytes * 8 // dtype.nbits
            global_indices = layout.get_global(local_index=start_i, spatial_index=emitter.current_thread)
            rewrite_map = {axis: as_expr(global_index) for axis, global_index in zip(axes, global_indices)}
            offset = rewrite(offset_expr, rewrite_map=rewrite_map)
            mask = rewrite(mask_expr, rewrite_map=rewrite_map)

            unit_bytes: int = gcd(vector_bytes, 16)
            unit_dtype: DataType = {1: uint8, 2: uint16, 4: uint32, 8: uint32x2, 16: uint32x4}[unit_bytes]
            num_units: int = vector_bytes // unit_bytes

            reg_ptr = emitter.declare_var("reg_ptr", ~unit_dtype, init=cast(~regs_buf[start_i], ~unit_dtype))
            mem_ptr = emitter.declare_var("mem_ptr", ~unit_dtype, init=cast(~smem_buf[offset], ~unit_dtype))
            dst_ptr, src_ptr = mem_ptr, reg_ptr
            with emitter.if_then(mask):
                with emitter.for_range(extent=num_units) as i:
                    emitter.buffer_store(buf=dst_ptr, indices=[i], value=src_ptr[i])
    else:
        with emitter.for_range(extent=register_tensor.local_size) as i:
            global_indices = layout.get_global(local_index=i, spatial_index=emitter.current_thread)
            rewrite_map = {axis: as_expr(global_index) for axis, global_index in zip(axes, global_indices)}
            offset = rewrite(offset_expr, rewrite_map=rewrite_map)
            mask = rewrite(mask_expr, rewrite_map=rewrite_map)
            with emitter.if_then(mask):
                emitter.buffer_store(buf=smem_buf, indices=[offset], value=regs_buf[i])


@register_emitter(LoadSharedInst, target=nvgpu_sm75)
class LoadSharedInstLdmatrixEmitter(BaseInstEmitter):
    """Emitter for LoadSharedInst that tries ldmatrix first, then falls back to generic loads."""

    def emit(self, inst: LoadSharedInst) -> None:
        register_tensor = inst.register_output
        shared_tensor = inst.shared_input
        dtype = register_tensor.dtype

        # Try ldmatrix
        config = _get_load_matrix_config(dtype, register_layout=register_tensor.layout)
        if config is not None:
            if _check_shared_alignment_and_contiguity(
                shared_tensor=shared_tensor,
                register_shape=register_tensor.shape,
                analysis=self.analysis,
                config_nbytes=config.nbytes,
                unit_shape_last=config.ldmatrix_layout.shape[-1],
            ):
                self._emit_ldmatrix(inst, config)
                return

        # Fallback to generic loads
        _emit_generic_load_shared(self, inst)

    def _emit_ldmatrix(self, inst: LoadSharedInst, config: LoadMatrixConfig) -> None:
        """Emit ldmatrix instruction (merged from cuda/ldmatrix.py emitter)."""
        tensor = inst.register_output
        shared_tensor = inst.shared_input
        layout = tensor.layout
        ldmatrix_layout = config.ldmatrix_layout

        lhs_layout: RegisterLayout = divide(layout, ldmatrix_layout)

        regs_buf = self.get_or_allocate_var(tensor)

        vector_size: int = gcd(lhs_layout.local_size, 4)
        num_vectors: int = lhs_layout.local_size // vector_size

        dtype = tensor.dtype

        # Get shared-space address for ldmatrix
        smem_base_addr = self.declare_var(
            "smem_addr", tp=int32, init=self.shared_tensor_shared_space_addr[shared_tensor]
        )

        # Get axes and byte offset from shared layout
        axes: list[Var] = index_vars(num_vars=len(shared_tensor.shape))
        byte_offset: Expr = shared_tensor.layout.byte_offset(*axes, nbytes=dtype.nbytes)

        with self.for_range(num_vectors, attr="u+") as vec_i:
            regs: list[Expr] = []
            for i in range(vector_size):
                regs.append(cast(~regs_buf[(vec_i * vector_size + i) * ldmatrix_layout.local_size], ~uint32))

            lane_id = self.current_thread % 32
            warp_id = self.current_thread // 32
            lhs_indices = vector(
                lhs_layout.get_global(local_index=vec_i * vector_size + lane_id // 8, spatial_index=warp_id)
            )
            rhs_indices = vector([lane_id % 8, 0])
            rhs_shape = vector(ldmatrix_layout.shape)
            shared_indices = list(lhs_indices * rhs_shape + rhs_indices)

            rewrite_map: Mapping[Node, Node] = {axis: index for axis, index in zip(axes, shared_indices)}
            byte_offset_rewritten = rewrite(byte_offset, rewrite_map=rewrite_map)
            smem_addr = smem_base_addr + byte_offset_rewritten

            self.append(ldmatrix(regs=regs, smem_addr=smem_addr, shared_space_addr=True, trans=config.trans))


@register_emitter(LoadSharedInst)
class LoadSharedInstGenericEmitter(BaseInstEmitter):
    """Generic emitter for LoadSharedInst (element-wise loads with vectorization)."""

    def emit(self, inst: LoadSharedInst) -> None:
        _emit_generic_load_shared(self, inst)


@register_emitter(StoreSharedInst, target=nvgpu_sm90)
class StoreSharedInstStmatrixEmitter(BaseInstEmitter):
    """Emitter for StoreSharedInst that tries stmatrix first, then falls back to generic stores.

    stmatrix requires sm_90 or higher.
    """

    def emit(self, inst: StoreSharedInst) -> None:
        shared_tensor = inst.inputs[0].as_shared_tensor()
        register_tensor = inst.inputs[1].as_register_tensor()
        dtype = register_tensor.dtype

        # Try stmatrix
        config = _get_store_matrix_config(dtype, register_layout=register_tensor.layout)
        if config is not None:
            if _check_shared_alignment_and_contiguity(
                shared_tensor=shared_tensor,
                register_shape=register_tensor.shape,
                analysis=self.analysis,
                config_nbytes=config.nbytes,
                unit_shape_last=config.stmatrix_layout.shape[-1],
            ):
                self._emit_stmatrix(inst, config)
                return

        # Fallback to generic stores
        _emit_generic_store_shared(self, inst)

    def _emit_stmatrix(self, inst: StoreSharedInst, config: StoreMatrixConfig) -> None:
        """Emit stmatrix instructions.

        Each thread computes its shared memory address from its position in the full
        register layout. The first local element (local_index=0) determines the address,
        since it corresponds to the first fp16 in the packed u32 register.
        """
        shared_tensor = inst.inputs[0].as_shared_tensor()
        register_tensor = inst.inputs[1].as_register_tensor()
        layout = register_tensor.layout

        regs_buf = self.get_or_allocate_var(register_tensor)

        stmatrix_layout = config.stmatrix_layout
        stmatrix_local_size = stmatrix_layout.local_size
        total_local = layout.local_size
        num_stmatrix_tiles = total_local // stmatrix_local_size
        vector_size: int = gcd(num_stmatrix_tiles, 4)
        num_vectors: int = num_stmatrix_tiles // vector_size

        dtype = register_tensor.dtype

        # Get shared-space address for stmatrix
        smem_base_addr = self.declare_var(
            "smem_addr", tp=int32, init=self.shared_tensor_shared_space_addr[shared_tensor]
        )

        # Get axes and byte offset from shared layout
        axes: list[Var] = index_vars(num_vars=len(shared_tensor.shape))
        byte_offset: Expr = shared_tensor.layout.byte_offset(*axes, nbytes=dtype.nbytes)

        with self.for_range(num_vectors, attr="u+") as vec_i:
            regs: list[Expr] = []
            for i in range(vector_size):
                local_idx = (vec_i * vector_size + i) * stmatrix_local_size
                regs.append(deref(cast(~regs_buf[local_idx], ~uint32)))

            # Compute shared memory address for stmatrix.
            # stmatrix address: T0 stores addr for row 0, T1 for row 1, ..., T7 for row 7.
            # For x2: T8-T15 provide addrs for the second matrix's rows 0-7, etc.
            # The address uses the same lhs/rhs decomposition as ldmatrix.
            # NOTE: this only works with non-swizzled shared layouts because stmatrix
            # writes 16 bytes sequentially from the given address.
            lane_id = self.current_thread % 32
            warp_id = self.current_thread // 32
            lhs_layout: RegisterLayout = divide(layout, stmatrix_layout)
            lhs_indices = vector(
                lhs_layout.get_global(local_index=vec_i * vector_size + lane_id // 8, spatial_index=warp_id)
            )
            rhs_indices = vector([lane_id % 8, 0])
            rhs_shape = vector(stmatrix_layout.shape)
            shared_indices = list(lhs_indices * rhs_shape + rhs_indices)

            rewrite_map: Mapping[Node, Node] = {axis: as_expr(idx) for axis, idx in zip(axes, shared_indices)}
            byte_offset_rewritten = rewrite(byte_offset, rewrite_map=rewrite_map)
            smem_addr = smem_base_addr + byte_offset_rewritten

            self.append(stmatrix(regs=regs, smem_addr=smem_addr, shared_space_addr=True, trans=config.trans))


@register_emitter(StoreSharedInst)
class StoreSharedInstGenericEmitter(BaseInstEmitter):
    """Generic emitter for StoreSharedInst (element-wise stores with vectorization)."""

    def emit(self, inst: StoreSharedInst) -> None:
        _emit_generic_store_shared(self, inst)
