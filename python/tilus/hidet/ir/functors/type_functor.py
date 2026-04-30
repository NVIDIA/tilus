# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=bad-staticmethod-argument
from tilus.hidet.ir.type import (
    DataType,
    FuncType,
    OpaqueType,
    PointerType,
    StringType,
    TensorPointerType,
    TensorType,
    VoidType,
)
from tilus.hidet.utils import same_list

from .base_functor import BaseFunctor, BaseRewriter, BaseVisitor


class TypeFunctor(BaseFunctor):
    def visit_dispatch(self, node):
        if isinstance(node, DataType):
            return self.visit_DataType(node)
        elif isinstance(node, TensorType):
            return self.visit_TensorType(node)
        elif isinstance(node, PointerType):
            return self.visit_PointerType(node)
        elif isinstance(node, TensorPointerType):
            return self.visit_TensorPointerType(node)
        elif isinstance(node, StringType):
            return self.visit_StringType(node)
        elif isinstance(node, VoidType):
            return self.visit_VoidType(node)
        elif isinstance(node, FuncType):
            return self.visit_FuncType(node)
        elif isinstance(node, OpaqueType):
            return self.visit_OpaqueType(node)
        else:
            return NotImplemented

    def visit_DataType(self, t: DataType):
        raise NotImplementedError()

    def visit_TensorType(self, t: TensorType):
        raise NotImplementedError()

    def visit_PointerType(self, t: PointerType):
        raise NotImplementedError()

    def visit_TensorPointerType(self, t: TensorPointerType):
        raise NotImplementedError()

    def visit_StringType(self, t: StringType):
        raise NotImplementedError()

    def visit_VoidType(self, t: VoidType):
        raise NotImplementedError()

    def visit_FuncType(self, t: FuncType):
        raise NotImplementedError()

    def visit_OpaqueType(self, t: OpaqueType):
        raise NotImplementedError()


class TypeVisitor(TypeFunctor, BaseVisitor):
    def visit_DataType(self, t: DataType):
        pass

    def visit_TensorType(self, t: TensorType):
        self.visit(t.dtype)
        self.visit(t.shape)

    def visit_PointerType(self, t: PointerType):
        self.visit(t.base_type)

    def visit_TensorPointerType(self, t: TensorPointerType):
        self.visit(t.tensor_type)

    def visit_StringType(self, t: StringType):
        pass

    def visit_VoidType(self, t: VoidType):
        pass

    def visit_FuncType(self, t: FuncType):
        self.visit(t.ret_type)
        for param_type in t.param_types:
            self.visit(param_type)

    def visit_OpaqueType(self, t: OpaqueType):
        pass


class TypeRewriter(TypeFunctor, BaseRewriter):
    def visit_DataType(self, t: DataType):
        return t

    def visit_TensorType(self, t: TensorType):
        dtype = self.visit(t.dtype)
        shape = self.visit(t.shape)
        if dtype == t.dtype and same_list(shape, t.shape):
            return t
        else:
            return TensorType(dtype, shape)

    def visit_PointerType(self, t: PointerType):
        base_type = self.visit(t.base_type)
        if base_type == t.base_type:
            return t
        else:
            return PointerType(base_type)

    def visit_TensorPointerType(self, t: TensorPointerType):
        tensor_type = self.visit(t.tensor_type)
        if tensor_type == t.tensor_type:
            return t
        else:
            return TensorPointerType(tensor_type)

    def visit_StringType(self, t: StringType):
        return t

    def visit_VoidType(self, t: VoidType):
        return t

    def visit_FuncType(self, t: FuncType):
        ret_type = self.visit(t.ret_type)
        param_types = [self.visit(param_type) for param_type in t.param_types]
        if ret_type == t.ret_type and same_list(param_types, t.param_types):
            return t
        else:
            return FuncType(param_types, ret_type)

    def visit_OpaqueType(self, t: OpaqueType):
        return t
