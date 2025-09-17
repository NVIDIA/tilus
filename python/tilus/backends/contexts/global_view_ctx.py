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

from hidet.ir.expr import Var
from hidet.ir.type import DataType

from tilus.backends.codegen import BaseEmitContext, FunctionCodegen, register_emit_context
from tilus.ir import GlobalLayout
from tilus.ir.tensor import GlobalTensor


@dataclass
class GlobalTensorView:
    ptr: Var
    dtype: DataType
    layout: GlobalLayout


@register_emit_context
class GlobalTensorViewContext(BaseEmitContext):
    """Context used to track the global tensor views that takes kernel parameters as ptr."""

    def __init__(self, codegen: FunctionCodegen):
        super().__init__(codegen)
        self.tensor2view: dict[GlobalTensor, GlobalTensorView] = {}

    def add_tensor_view(self, tensor: GlobalTensor, ptr: Var, layout: GlobalLayout) -> None:
        assert tensor not in self.tensor2view
        self.tensor2view[tensor] = GlobalTensorView(ptr, tensor.dtype, layout)
