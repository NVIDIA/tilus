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

from hidet.ir.expr import Expr, Var
from hidet.ir.type import DataType
from tilus.backends.codegen import BaseInstEmitter
from tilus.backends.codegen import register_emitter
from tilus.backends.contexts import GlobalTensorViewContext, GlobalTensorView
from tilus.ir.tensor import GlobalTensor, SharedTensor
from tilus.ir.instructions.cuda.cp_async_tensor import (
    CopyAsyncTensorGlobalToSharedInst
)
from tilus.target import nvgpu_sm90
from tilus.extensions.hidet.ir.primitives.cuda.tensor_map import encode_tensor_map, TensorMapSwizzle
from tilus.extensions.hidet.ir.primitives.cuda.copy_async_tensor import cp_async_tensor_global_to_shared
from tilus.ir.utils.lineardec import decompose_linear, LinearDecompositionError


@dataclass(frozen=True, eq=False)
class GlobalTensorInfo:
    ptr: Expr
    shape: tuple[Expr, ...]
    strides: tuple[Expr, ...]

@dataclass(frozen=True, eq=False)
class SharedTensorInfo:
    addr: Expr
    swizzle: TensorMapSwizzle


class CopyAsyncTensorBaseEmitter(BaseInstEmitter):
    def resolve_global_tensor_info(self, global_tensor: GlobalTensor) -> GlobalTensorInfo:
        ctx: GlobalTensorViewContext = self.contexts[GlobalTensorViewContext]

        if global_tensor not in ctx.tensor2view:
            raise ValueError('TMA only supports global tensors created by global_view with pointer as kernel parameter')

        view = ctx.tensor2view[global_tensor]

        try:
            coefficients = decompose_linear(view.layout.offset, coordinates=view.layout.axes)
        except LinearDecompositionError:
            raise ValueError('TMA only supports strided global tensors')

        if len(coefficients) != len(view.layout.axes):
            raise ValueError('TMA only supports strided global tensors without constant offset')

        return GlobalTensorInfo(ptr=view.ptr, shape=view.layout.shape, strides=tuple(coefficients))


    def resolve_shared_tensor_info(self, shared_tensor: SharedTensor) -> SharedTensorInfo:
        raise NotImplementedError()

    def create_tensor_map(self, global_info: GlobalTensorInfo, shared_info: SharedTensorInfo, dtype: DataType) -> Var:
        raise NotImplementedError()


@register_emitter(CopyAsyncTensorGlobalToSharedInst, target=nvgpu_sm90)
class CopyAsyncTensorGlobalToSharedInstEmitter(CopyAsyncTensorBaseEmitter):
    def emit(self, inst: CopyAsyncTensorGlobalToSharedInst) -> None:
        global_tensor: GlobalTensor = inst.inputs[1].as_global_tensor()
        shared_tensor: SharedTensor = inst.inputs[0].as_shared_tensor()
        assert global_tensor.dtype == shared_tensor.dtype
        dtype: DataType = global_tensor.dtype

        global_tensor_info: GlobalTensorInfo = self.resolve_global_tensor_info(global_tensor)
        shared_tensor_info: SharedTensorInfo = self.resolve_shared_tensor_info(shared_tensor)

        shared_addr = self.shared_tensor_shared_space_addr[shared_tensor]
        tensor_map = self.create_tensor_map(global_tensor_info, shared_tensor_info, dtype)
        tensor_coords = inst.offsets
        self.append(
            cp_async_tensor_global_to_shared(
                dst=shared_addr,
                src=self.tensor2var[global_tensor],
                tensor_map=tensor_map,
                coords=tensor_coords,
                mbarrier=inst.mbarrier,
                cache_policy=inst.cache_policy,
            )
        )
