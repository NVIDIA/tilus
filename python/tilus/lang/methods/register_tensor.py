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
from __future__ import annotations

import itertools
from typing import Sequence

from hidet.ir.expr import Expr, Var
from hidet.ir.type import DataType

from tilus.ir.builders import StmtBuilder
from tilus.ir.tensor import RegisterTensor


class RegisterTensorWithMethods(RegisterTensor):
    def __init__(self, tensor: RegisterTensor, builder: StmtBuilder):
        self.tensor: RegisterTensor = tensor
        self.builder: StmtBuilder = builder

    def __neg__(self) -> RegisterTensor:
        return self.builder.neg(self.tensor)

    def __add__(self, other: RegisterTensorWithMethods | RegisterTensor | int | float | Expr) -> RegisterTensor:
        if isinstance(other, RegisterTensorWithMethods):
            other = other.tensor
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(
                dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other)
            )
        return self.builder.add(self.tensor, other)

    def __sub__(self, other: RegisterTensorWithMethods | RegisterTensor | int | float | Expr) -> RegisterTensor:
        if isinstance(other, RegisterTensorWithMethods):
            other = other.tensor
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(
                dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other)
            )
        return self.builder.sub(self.tensor, other)

    def __mul__(self, other: RegisterTensorWithMethods | RegisterTensor | int | float | Expr) -> RegisterTensor:
        if isinstance(other, RegisterTensorWithMethods):
            other = other.tensor
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(
                dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other)
            )
        return self.builder.mul(self.tensor, other)

    def __truediv__(self, other: RegisterTensorWithMethods | RegisterTensor | int | float | Expr) -> RegisterTensor:
        if isinstance(other, RegisterTensorWithMethods):
            other = other.tensor
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(
                dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other)
            )
        return self.builder.div(self.tensor, other)

    def __ge__(self, other: RegisterTensorWithMethods | RegisterTensor | int | float | Expr) -> RegisterTensor:
        if isinstance(other, RegisterTensorWithMethods):
            other = other.tensor
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(
                dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other)
            )
        return self.builder.greater_equal(self.tensor, other)

    def __le__(self, other: RegisterTensorWithMethods | RegisterTensor | int | float | Expr) -> RegisterTensor:
        if isinstance(other, RegisterTensorWithMethods):
            other = other.tensor
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(
                dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other)
            )
        return self.builder.less_equal(self.tensor, other)

    def __gt__(self, other: RegisterTensorWithMethods | RegisterTensor | int | float | Expr) -> RegisterTensor:
        if isinstance(other, RegisterTensorWithMethods):
            other = other.tensor
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(
                dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other)
            )
        return self.builder.greater_than(self.tensor, other)

    def __lt__(self, other: RegisterTensorWithMethods | RegisterTensor | int | float | Expr) -> RegisterTensor:
        if isinstance(other, RegisterTensorWithMethods):
            other = other.tensor
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(
                dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other)
            )
        return self.builder.less_than(self.tensor, other)

    def __eq__(self, other: RegisterTensorWithMethods | RegisterTensor | int | float | Expr) -> RegisterTensor:
        if isinstance(other, RegisterTensorWithMethods):
            other = other.tensor
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(
                dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other)
            )
        return self.builder.equal(self.tensor, other)

    def __ne__(self, other: RegisterTensorWithMethods | RegisterTensor | int | float | Expr) -> RegisterTensor:
        if isinstance(other, RegisterTensorWithMethods):
            other = other.tensor
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(
                dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other)
            )
        return self.builder.not_equal(self.tensor, other)

    def item(self) -> Var:
        return self.builder.tensor_item_value(self.tensor)

    def squeeze(self, dim: int | Sequence[int]) -> RegisterTensor:
        return self.builder.squeeze(self.tensor, dim)

    def unsqueeze(self, dim: int | Sequence[int]) -> RegisterTensor:
        return self.builder.unsqueeze(self.tensor, dim)

    def transpose(self) -> RegisterTensor:
        return self.builder.transpose(self.tensor)

    def to(self, dtype: DataType) -> RegisterTensor:
        return self.builder.cast(self.tensor, dtype=dtype)

    def tolist(self) -> Expr | list:
        if len(self.tensor.shape) == 0:
            return self.builder.tensor_item_value(self.tensor)
        else:
            ret = []
            for indices in itertools.product(*(range(s) for s in self.tensor.shape)):
                ret.append(
                    self.builder.tensor_item_value(
                        self.builder.slice_register(self.tensor, offsets=indices, slice_dims=[], slice_shape=[])
                    )
                )
            return ret
