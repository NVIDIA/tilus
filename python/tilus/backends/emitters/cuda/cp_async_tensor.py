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

from hidet.ir.expr import Expr
from hidet.ir.type import DataType
from tilus.backends.codegen import BaseInstEmitter
from tilus.backends.codegen import register_emitter
from tilus.ir.tensor import GlobalTensor, SharedTensor
from tilus.ir.instructions.cuda.cp_async_tensor import (
    CopyAsyncTensorGlobalToSharedInst
)
from tilus.target import nvgpu_sm90
from tilus.extensions.hidet.ir.primitives.cuda.tensor_map import encode_tensor_map, TensorMapSwizzle


@dataclass(frozen=True, eq=False)
class GlobalTensorInfo:
    ptr: Expr
    shape: tuple[Expr, ...]
    strides: tuple[Expr, ...]

@dataclass(frozen=True, eq=False)
class SharedTensorInfo:
    addr: Expr
    swizzle: TensorMapSwizzle



class CopyAsyncTensorBaseEmitter(BaseInstcheEmitter):
    def resolve_global_tensor_info(self, global_tensor: GlobalTensor) -> GlobalTensorInfo:
        raise NotImplementedError()

    def resolve_shared_tensor_info(self, shared_tensor: SharedTensor) -> SharedTensorInfo:
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



