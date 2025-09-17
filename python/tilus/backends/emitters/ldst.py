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
from typing import Optional, Union

from hidet.ir.dtypes import boolean, uint8, uint16, uint32
from hidet.ir.expr import Var, as_expr, cast, if_then_else
from hidet.ir.type import DataType

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.extensions.hidet.ir.dtypes.vector import uint32x2, uint32x4
from tilus.extensions.hidet.ir.expr import index_vars
from tilus.extensions.hidet.ir.tools.rewriter import rewrite
from tilus.ir.analyzers.grid_analyzer import TensorInfo, analyze_grid
from tilus.ir.instructions import (
    LoadGlobalGenericInst,
    LoadSharedGenericInst,
    StoreGlobalGenericInst,
    StoreSharedGenericInst,
)
from tilus.ir.layout import RegisterLayout
from tilus.utils import gcd


class LoadStoreInstBaseEmitter(BaseInstEmitter):
    def analyze_vectorization(
        self,
        inst: Union[LoadGlobalGenericInst, StoreGlobalGenericInst, LoadSharedGenericInst, StoreSharedGenericInst],
    ) -> Optional[tuple[int, int]]:
        """
        Analyze the applicable vectorization of the load/store instruction to global or shared memory.

        Give a tile of data to be loaded or stored, we try to find the dimension along which we can load/store multiple
        elements at once. We can split the tile of elements into a tile of vectors of elements. And the elements in each
        vector must satisfy:
        1. The elements in each vector must be hold by a single thread.
        2. The vector elements are contiguous in memory.
        3. The vector elements are contiguous in the local storage of the thread.
        4. The mask of loading/storing the vector elements must be identical for all elements in the vector.

        We will determine the dimension along which we can vectorize the load/store, and the number of bytes in each
        vector. We support all normal data types (like 8-bit, 16-bit, or 32-bit data) and all sub-byte data types (like
        1-bit, 2-bit, ... and 7-bit data type).
        """
        # get the register value that is going to be stored or loaded to.
        if isinstance(inst, (LoadGlobalGenericInst, LoadSharedGenericInst)):
            regs_tensor = inst.register_output
        elif isinstance(inst, (StoreGlobalGenericInst, StoreSharedGenericInst)):
            regs_tensor = inst.register_input
        else:
            raise NotImplementedError()

        dtype: DataType = regs_tensor.dtype
        shape: tuple[int, ...] = regs_tensor.layout.shape
        layout: RegisterLayout = regs_tensor.layout

        # analyze the offset and mask's value information (e.g., divisibility, constancy, etc.)
        analysis = self.analysis
        offset_info = analyze_grid(shape=shape, axes=inst.axes, analysis=analysis, expr=inst.offset)
        mask_info = analyze_grid(shape=shape, axes=inst.axes, analysis=analysis, expr=inst.mask)

        # analyze the register layout so that we can know how the elements are distributed stored in threads
        axes = index_vars(len(layout.shape))
        expr = layout.get_local(global_indices=axes)
        layout_info: TensorInfo = analyze_grid(shape=layout.shape, axes=axes, analysis=analysis, expr=as_expr(expr))

        # enumerate each dimension and check whether we can vectorize on that dimension
        for i in range(len(regs_tensor.shape)):
            max_vector_elements = gcd(  # to be eligible for vectorized loading, the elements must:
                offset_info[i].divisibility,  # the offset must be divisible by the vector size
                offset_info[i].continuity,  # contiguous in global/shared memory (cond. 2)
                layout_info[i].continuity,  # contiguous in the local storage of the thread (cond. 1 and 3)
                mask_info[i].constancy,  # the mask must be the same for all elements in the vector (cond. 4)
                layout.local_size,  # the local storage must be able to be divided into multiple such vectors
            )
            if max_vector_elements > 1 and max_vector_elements * dtype.nbits % 8 == 0:
                # the vector elements must be able to be represented by multiple bytes
                vectorize_dimension = i
                vector_bytes = max_vector_elements * dtype.nbits // 8
                return vectorize_dimension, vector_bytes
        else:
            # failed to use vectorized loading
            return None


@register_emitter(LoadSharedGenericInst)
@register_emitter(LoadGlobalGenericInst)
@register_emitter(StoreSharedGenericInst)
@register_emitter(StoreGlobalGenericInst)
class LoadGlobalGenericInstEmitter(LoadStoreInstBaseEmitter):
    def emit(
        self, inst: LoadGlobalGenericInst | LoadSharedGenericInst | StoreGlobalGenericInst | StoreSharedGenericInst
    ) -> None:
        if isinstance(inst, (LoadGlobalGenericInst, LoadSharedGenericInst)):
            tensor = inst.register_output
        else:
            tensor = inst.register_input
        dtype: DataType = tensor.dtype
        layout: RegisterLayout = tensor.layout
        regs_buf: Var = self.get_or_allocate_var(tensor=tensor)

        vectorization: Optional[tuple[int, int]] = self.analyze_vectorization(inst)
        if vectorization:
            vectorize_dimension, vector_bytes = vectorization
            total_nbytes = layout.local_size * dtype.nbits // 8
            with self.for_range(extent=total_nbytes // vector_bytes) as vec_i:
                # the first local element of this vector
                start_i = vec_i * vector_bytes * 8 // dtype.nbits

                # the corresponding global indices in the register tensor
                global_indices = layout.get_global(local_index=start_i, spatial_index=self.current_thread)

                # the offset and mask of this vector
                rewrite_map = {axis: as_expr(global_index) for axis, global_index in zip(inst.axes, global_indices)}
                offset = rewrite(inst.offset, rewrite_map=rewrite_map)
                mask = rewrite(inst.mask, rewrite_map=rewrite_map) if inst.mask is not None else boolean.true

                # the vector_nbytes might be number like 6 bytes, divide it into units
                unit_bytes: int = gcd(vector_bytes, 16)  # each unit could be either 1, 2, or 4 bytes
                unit_dtype: DataType = {1: uint8, 2: uint16, 4: uint32, 8: uint32x2, 16: uint32x4}[unit_bytes]
                num_units: int = vector_bytes // unit_bytes

                reg_ptr = self.declare_var("reg_ptr", ~unit_dtype, init=cast(~regs_buf[start_i], ~unit_dtype))
                mem_ptr = self.declare_var("mem_ptr", ~unit_dtype, init=cast(~inst.ptr[offset], ~unit_dtype))
                if isinstance(inst, (LoadGlobalGenericInst, LoadSharedGenericInst)):
                    dst_ptr, src_ptr = reg_ptr, mem_ptr
                    with self.if_then(mask):
                        with self.for_range(extent=num_units) as i:
                            self.buffer_store(buf=dst_ptr, indices=[i], value=src_ptr[i])
                    with self.otherwise():
                        with self.for_range(extent=num_units) as i:
                            self.buffer_store(buf=dst_ptr, indices=[i], value=unit_dtype.zero)
                else:
                    dst_ptr, src_ptr = mem_ptr, reg_ptr
                    with self.if_then(mask):
                        with self.for_range(extent=num_units) as i:
                            self.buffer_store(buf=dst_ptr, indices=[i], value=src_ptr[i])

        else:
            with self.for_range(extent=tensor.local_size) as i:
                global_indices = layout.get_global(local_index=i, spatial_index=self.current_thread)
                rewrite_map = {axis: as_expr(global_index) for axis, global_index in zip(inst.axes, global_indices)}
                offset = rewrite(inst.offset, rewrite_map=rewrite_map)
                mask = rewrite(inst.mask, rewrite_map=rewrite_map) if inst.mask is not None else boolean.true
                if isinstance(inst, (LoadGlobalGenericInst, LoadSharedGenericInst)):
                    self.buffer_store(buf=regs_buf, indices=[i], value=if_then_else(mask, inst.ptr[offset], dtype.zero))
                else:
                    with self.if_then(mask):
                        self.buffer_store(buf=inst.ptr, indices=[offset], value=regs_buf[i])
