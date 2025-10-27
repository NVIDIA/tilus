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
from hidet.ir.expr import Var
from tilus.ir.builders import StmtBuilder
from tilus.ir.tensor import SharedTensor
from tilus.lang.methods.exception import TensorMethodError


class SharedTensorWithMethods(SharedTensor):
    def __init__(self, tensor: SharedTensor, builder: StmtBuilder):
        super().__init__(tensor.dtype, tensor.shape, tensor.optional_layout)
        self.tensor: SharedTensor = tensor
        self.builder: StmtBuilder = builder

    def item_ptr(self) -> Var:
        return self.builder.tensor_item_ptr(self.tensor, space="generic")

    def item(self) -> Var:
        return self.builder.tensor_item_value(self.tensor)

    def permute(self, dims: tuple[int, ...]) -> SharedTensor:
        if set(dims) != set(range(len(self.tensor.shape))):
            raise TensorMethodError(f"Dims must be a permutation of {range(len(self.tensor.shape))}, got {dims}")
        return self.builder.permute_shared(self.tensor, dims)

    def transpose(self) -> SharedTensor:
        return self.builder.permute_shared(self.tensor, dims=[1, 0])
