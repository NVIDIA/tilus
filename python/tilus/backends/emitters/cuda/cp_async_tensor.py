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
import functools
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from hidet.ir import logical_or
from hidet.ir.dtypes import uint32, uint64
from hidet.ir.expr import Expr, Var, as_expr, cast
from hidet.ir.tools import simplify
from hidet.ir.type import DataType, PointerType, TensorType, sizeof

from tilus import SharedLayout
from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.backends.contexts import GlobalTensorViewContext, InvariantTrackingContext
from tilus.extensions.hidet.ir.expr import index_vars
from tilus.extensions.hidet.ir.primitives.cuda.copy_async_tensor import (
    cp_async_tensor_commit_group,
    cp_async_tensor_global_to_shared,
    cp_async_tensor_shared_to_global,
    cp_async_tensor_wait_group,
)
from tilus.extensions.hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared
from tilus.extensions.hidet.ir.primitives.cuda.mbarrier import mbarrier_expect_tx_cta_shared
from tilus.extensions.hidet.ir.primitives.cuda.tensor_map import (
    CUtensorMapType,
    TensorMapDataType,
    TensorMapFloatOOBFill,
    TensorMapInterleave,
    TensorMapL2Promotion,
    TensorMapSwizzle,
    encode_tensor_map,
)
from tilus.extensions.hidet.ir.tools import rewrite
from tilus.ir import GlobalLayout
from tilus.ir.instructions.cuda.cp_async_tensor import (
    CopyAsyncTensorCommitGroupInst,
    CopyAsyncTensorGlobalToSharedInst,
    CopyAsyncTensorSharedToGlobalInst,
    CopyAsyncTensorWaitGroupInst,
)
from tilus.ir.tensor import GlobalTensor, SharedTensor
from tilus.ir.utils.lineardec import LinearDecompositionError, decompose_linear
from tilus.ir.utils.veceval import vectorized_evaluate
from tilus.target import nvgpu_sm90
from tilus.utils import prod


@dataclass(frozen=True, eq=False)
class GlobalTensorInfo:
    ptr: Expr
    shape: tuple[Expr, ...]
    strides: tuple[Expr, ...]


@dataclass(frozen=True, eq=False)
class SharedTensorInfo:
    addr: Expr
    shape: tuple[int, ...]
    swizzle: TensorMapSwizzle


def cast_ptr_if_needed(ptr: Var, dtype: DataType) -> Expr:
    if isinstance(ptr.type, PointerType) and ptr.type.base_type == dtype:
        return ptr
    else:
        return cast(ptr, ~dtype)


def get_strides(shape: Sequence[int]) -> tuple[int, ...]:
    strides = [1]
    for extent in reversed(shape[1:]):
        strides.append(strides[-1] * extent)
    return tuple(reversed(strides))


def log2(x: int) -> int:
    if x == 1:
        return 0
    elif (x & 1) == 0:
        return 1 + log2(x >> 1)
    else:
        raise ValueError("x is not a power of 2")


@functools.cache
def get_offset_grid_of_swizzled_layout(
    dtype_nbits: int, shape: tuple[int, ...], swizzle: TensorMapSwizzle
) -> Optional[np.ndarray]:
    range_indices: list[np.ndarray] = []
    for dim, extent in enumerate(shape):
        range_indices.append(np.arange(extent, dtype=np.int32))
    grid: tuple[np.ndarray, ...] = np.meshgrid(*range_indices, indexing="ij")
    axes: list[Var] = index_vars(len(shape))
    strides = get_strides(shape)
    offset = as_expr(sum(axes[i] * strides[i] for i in range(len(shape))))

    # offset regards the original data type pointer
    offset_grid: np.ndarray = vectorized_evaluate(expr=offset, var2value={axis: grid[i] for i, axis in enumerate(axes)})

    # c: bit count
    # d: start bit offset that will be applied the bitwise-xor
    # r: start bit offset that will be used to perform the bitwise-xor against
    # we use bit-address for genericity
    Swizzle = namedtuple("Swizzle", ["c", "d", "r"])

    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-swizzling-modes
    swizzles: list[Swizzle] = []
    if swizzle == TensorMapSwizzle.NONE:
        pass
    elif swizzle == TensorMapSwizzle.B32:
        swizzles.append(Swizzle(c=1, d=log2(128), r=log2(1024)))
    elif swizzle == TensorMapSwizzle.B64:
        swizzles.append(Swizzle(c=2, d=log2(128), r=log2(1024)))
    elif swizzle == TensorMapSwizzle.B128:
        swizzles.append(Swizzle(c=3, d=log2(128), r=log2(1024)))
    elif swizzle == TensorMapSwizzle.B128_ATOM_32B:
        swizzles.append(Swizzle(c=3, d=log2(256), r=log2(2048)))
    elif swizzle == TensorMapSwizzle.B128_ATOM_32B_FLIP_8B:
        swizzles.append(Swizzle(c=3, d=log2(256), r=log2(2048)))
        swizzles.append(Swizzle(c=1, d=log2(64), r=log2(512)))
    elif swizzle == TensorMapSwizzle.B128_ATOM_64B:
        swizzles.append(Swizzle(c=3, d=log2(512), r=log2(4096)))
    else:
        # unsupported swizzle
        return None

    # bit-offset
    offset_grid = offset_grid * dtype_nbits

    # apply swizzling
    for swizzle in swizzles:
        offset_grid = offset_grid ^ (((offset_grid >> swizzle.r) & ((1 << swizzle.c) - 1)) << swizzle.d)

    # convert back to dtype pointer offset
    if np.any(offset_grid & (dtype_nbits - 1)):
        # the offset is not aligned to the data type size
        return None

    offset_grid = offset_grid // dtype_nbits
    return offset_grid


class CopyAsyncTensorBaseEmitter(BaseInstEmitter):
    def resolve_global_tensor_info(
        self, global_tensor: GlobalTensor, offsets: Sequence[Expr], dims: Sequence[int]
    ) -> GlobalTensorInfo:
        ctx: GlobalTensorViewContext = GlobalTensorViewContext.current()

        # get the global tensor view
        if global_tensor not in ctx.tensor2view:
            raise ValueError("TMA only supports global tensors created by global_view with pointer as kernel parameter")

        view = ctx.tensor2view[global_tensor]

        # process the indexing
        assert len(offsets) == len(global_tensor.shape)
        layout: GlobalLayout = global_tensor.layout
        indexing_dims = [dim for dim in range(len(offsets)) if dim not in dims]
        remap: dict[Var, Expr] = {layout.axes[dim]: offsets[dim] for dim in indexing_dims}
        offset = rewrite(layout.offset, remap)

        # get the coordinates and coefficients
        coordinates = [layout.axes[dim] for dim in dims]
        try:
            coefficients = decompose_linear(offset, coordinates=coordinates)
        except LinearDecompositionError:
            raise ValueError("TMA only supports strided global tensors")
        coefficients = simplify(coefficients)

        # get the starting address of the tensor box that is being copied
        dtype = global_tensor.dtype
        constant_offset = simplify(coefficients[-1])
        ptr = cast_ptr_if_needed(view.ptr, dtype) + constant_offset
        shape = tuple(extent for i, extent in enumerate(global_tensor.shape) if i in dims)
        strides = tuple(coefficients[:-1])

        # rewrite the ptr, shape, and strides to grid-invariant form (so that they can be used in host code)
        ctx: InvariantTrackingContext = InvariantTrackingContext.current()
        ptr = ctx.rewrite_to_grid_invariant(ptr)
        shape = tuple(ctx.rewrite_to_grid_invariant(s) for s in shape)
        strides = tuple(ctx.rewrite_to_grid_invariant(s) for s in strides)

        return GlobalTensorInfo(ptr=ptr, shape=shape, strides=strides)

    def resolve_shared_tensor_info(self, shared_tensor: SharedTensor) -> SharedTensorInfo:
        range_indices: list[np.ndarray] = []
        for dim, extent in enumerate(shared_tensor.shape):
            range_indices.append(np.arange(extent, dtype=np.int32))
        grid = np.meshgrid(*range_indices, indexing="ij")
        layout: SharedLayout = shared_tensor.layout

        offset_grid: np.ndarray = vectorized_evaluate(
            expr=layout.offset, var2value={axis: grid[i] for i, axis in enumerate(layout.axes)}
        )
        for swizzle in [
            TensorMapSwizzle.NONE,
            TensorMapSwizzle.B32,
            TensorMapSwizzle.B64,
            TensorMapSwizzle.B128,
            TensorMapSwizzle.B128_ATOM_32B,
            TensorMapSwizzle.B128_ATOM_32B_FLIP_8B,
            TensorMapSwizzle.B128_ATOM_64B,
        ]:
            swizzled_offset_grid = get_offset_grid_of_swizzled_layout(
                dtype_nbits=shared_tensor.dtype.nbits, shape=shared_tensor.shape, swizzle=swizzle
            )
            print(f"swizzle: {swizzle}")
            print(f"swizzled_offset_grid: {swizzled_offset_grid[:16]}")
            if swizzled_offset_grid is not None and np.array_equal(offset_grid, swizzled_offset_grid):
                return SharedTensorInfo(
                    addr=self.shared_tensor_shared_space_addr[shared_tensor], shape=shared_tensor.shape, swizzle=swizzle
                )
        raise NotImplementedError(
            "The shared tensor layout is not supported by TMA: \n"
            + f"Shared tensor: {shared_tensor.dtype.name}{list(shared_tensor.shape)}\n"
            + layout.visualize()
        )

    def declare_host_buffer(self, name: str, dtype: DataType, shape: Sequence[int]) -> Var:
        from hidet.ir.layout import strided_layout

        return self.host_builder.declare_var(
            name=name, tp=TensorType(dtype=dtype, shape=shape, layout=strided_layout(shape=shape))
        )

    def create_tensor_map(self, global_info: GlobalTensorInfo, shared_info: SharedTensorInfo, dtype: DataType) -> Var:
        tensor_map = self.host_builder.declare_var(name="tma_tensor_map", tp=CUtensorMapType)

        # rank
        rank = len(global_info.shape)

        # global shape
        shape_buf = self.declare_host_buffer(name="tma_shape", dtype=uint64, shape=[rank])
        rev_global_shape = list(reversed(global_info.shape))
        for i in range(rank):
            self.host_builder.buffer_store(shape_buf, indices=[i], value=as_expr(rev_global_shape[i]))

        # global strides
        strides_buf = self.declare_host_buffer(name="tma_strides", dtype=uint64, shape=[rank - 1])
        rev_global_strides = list(reversed(global_info.strides))
        self.host_builder.assertion(
            cond=rev_global_strides[0] == 1, msg="The last dimension of the global tensor must be contiguous"
        )
        for i in range(rank - 1):
            self.host_builder.buffer_store(
                strides_buf, indices=[i], value=as_expr(rev_global_strides[i + 1]) * sizeof(dtype)
            )

        # box shape
        box_shape_buf = self.declare_host_buffer(name="tma_box_shape", dtype=uint32, shape=[rank])
        rev_box_shape = list(reversed(shared_info.shape))
        for i in range(rank):
            self.host_builder.buffer_store(box_shape_buf, indices=[i], value=as_expr(rev_box_shape[i]))

        # element-wise strides
        elem_strides_buf = self.declare_host_buffer(name="tma_elem_strides", dtype=uint32, shape=[rank])
        for i in range(rank):
            self.host_builder.buffer_store(elem_strides_buf, indices=[i], value=uint32.one)

        # encode the tensor map
        self.host_builder.append(
            encode_tensor_map(
                tensor_map=~tensor_map,
                dtype=TensorMapDataType.from_dtype(dtype),
                rank=uint32(rank),
                tensor_ptr=global_info.ptr,
                shape=shape_buf,
                strides=strides_buf,
                box_shape=box_shape_buf,
                elem_strides=elem_strides_buf,
                interleave=TensorMapInterleave.NONE,
                swizzle=shared_info.swizzle,
                l2_promotion=TensorMapL2Promotion.B128,
                oob_fill=TensorMapFloatOOBFill.NONE,
            )
        )

        # ensure the tensor map is passed to the kernel
        self.append_extra_param(tensor_map)

        return tensor_map


@register_emitter(CopyAsyncTensorGlobalToSharedInst, target=nvgpu_sm90)
class CopyAsyncTensorGlobalToSharedInstEmitter(CopyAsyncTensorBaseEmitter):
    def emit(self, inst: CopyAsyncTensorGlobalToSharedInst) -> None:
        global_tensor: GlobalTensor = inst.inputs[1].as_global_tensor()
        shared_tensor: SharedTensor = inst.inputs[0].as_shared_tensor()
        assert global_tensor.dtype == shared_tensor.dtype
        dtype: DataType = global_tensor.dtype

        global_tensor_info: GlobalTensorInfo = self.resolve_global_tensor_info(
            global_tensor, offsets=inst.offsets, dims=inst.dims
        )

        shared_tensor_info: SharedTensorInfo = self.resolve_shared_tensor_info(shared_tensor)

        shared_addr = self.shared_tensor_shared_space_addr[shared_tensor]
        tensor_map = self.create_tensor_map(global_tensor_info, shared_tensor_info, dtype)
        tensor_coords = inst.offsets
        transaction_bytes = prod(shared_tensor.shape) * dtype.nbytes
        with self.if_then(logical_or(self.current_num_threads == 1, self.current_thread == 0)):
            barrier_addr = self.declare_var("barrier_addr", uint32, init=cvta_generic_to_shared(inst.mbarrier))
            self.append(mbarrier_expect_tx_cta_shared(mbarrier_addr=barrier_addr, transaction_bytes=transaction_bytes))
            self.append(
                cp_async_tensor_global_to_shared(
                    dst=shared_addr,
                    src_tensor_map=~tensor_map,
                    coords=list(reversed(tensor_coords)),
                    mbarrier=barrier_addr,
                    cta_group=None,
                    cache_policy=inst.cache_policy,
                )
            )


@register_emitter(CopyAsyncTensorSharedToGlobalInst, target=nvgpu_sm90)
class CopyAsyncTensorSharedToGlobalInstEmitter(CopyAsyncTensorBaseEmitter):
    def emit(self, inst: CopyAsyncTensorSharedToGlobalInst) -> None:
        global_tensor: GlobalTensor = inst.inputs[0].as_global_tensor()
        shared_tensor: SharedTensor = inst.inputs[1].as_shared_tensor()
        assert global_tensor.dtype == shared_tensor.dtype
        dtype: DataType = global_tensor.dtype

        global_tensor_info: GlobalTensorInfo = self.resolve_global_tensor_info(
            global_tensor, offsets=inst.offsets, dims=inst.dims
        )

        shared_tensor_info: SharedTensorInfo = self.resolve_shared_tensor_info(shared_tensor)

        shared_addr = self.shared_tensor_shared_space_addr[shared_tensor]
        tensor_map = self.create_tensor_map(global_tensor_info, shared_tensor_info, dtype)
        tensor_coords = inst.offsets
        with self.if_then(logical_or(self.current_num_threads == 1, self.current_thread == 0)):
            self.append(
                cp_async_tensor_shared_to_global(
                    dst_tensor_map=~tensor_map,
                    src=shared_addr,
                    coords=list(reversed(tensor_coords)),
                    cache_policy=inst.cache_policy,
                )
            )


@register_emitter(CopyAsyncTensorCommitGroupInst, target=nvgpu_sm90)
class CopyAsyncCommitGroupInstEmitter(BaseInstEmitter):
    def emit(self, inst: CopyAsyncTensorCommitGroupInst) -> None:
        self.append(cp_async_tensor_commit_group())


@register_emitter(CopyAsyncTensorWaitGroupInst, target=nvgpu_sm90)
class CopyAsyncWaitGroupInstEmitter(BaseInstEmitter):
    def emit(self, inst: CopyAsyncTensorWaitGroupInst) -> None:
        self.append(cp_async_tensor_wait_group(inst.n))
