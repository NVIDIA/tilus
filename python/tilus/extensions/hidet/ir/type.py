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
from hidet.ir.type import BaseType, DataType, FuncType, OpaqueType, PointerType, TensorPointerType, TensorType, VoidType


def is_addressable(tp_or_var):
    from hidet.ir.expr import Var

    if isinstance(tp_or_var, Var):
        tp = tp_or_var.type
    else:
        tp = tp_or_var
    return isinstance(tp, (PointerType, TensorPointerType, TensorType))


def get_base_type(tp: BaseType) -> BaseType:
    if isinstance(tp, PointerType):
        return tp.base_type
    elif isinstance(tp, TensorPointerType):
        return tp.tensor_type.dtype
    elif isinstance(tp, TensorType):
        return tp.dtype
    else:
        assert False


def type_equal(lhs: BaseType, rhs: BaseType) -> bool:
    """
    Check whether the two types are equal or not.

    Parameters
    ----------
    lhs: BaseType
        The first type to compare.
    rhs: BaseType
        The second type to compare.

    Returns
    -------
    ret: bool
        Whether the two types are equal or not.
    """
    if type(lhs) is not type(rhs):
        return False
    if isinstance(lhs, DataType) and isinstance(rhs, DataType):
        return lhs.name == rhs.name
    elif isinstance(lhs, PointerType) and isinstance(rhs, PointerType):
        return type_equal(lhs.base_type, rhs.base_type)
    elif isinstance(lhs, VoidType) and isinstance(rhs, VoidType):
        return True
    elif isinstance(lhs, TensorPointerType) and isinstance(rhs, TensorPointerType):
        return type_equal(lhs.tensor_type, rhs.tensor_type)
    elif isinstance(lhs, TensorType) and isinstance(rhs, TensorType):
        from hidet.ir.expr import is_constant

        if not type_equal(lhs.dtype, rhs.dtype):
            return False
        if len(lhs.shape) != len(rhs.shape):
            return False
        for a, b in zip(lhs.shape, rhs.shape):
            if is_constant(a) ^ is_constant(b):
                return False
            elif is_constant(a) and is_constant(b):
                if int(a) != int(b):
                    return False
            else:
                # we do not have equivalence checking for symbolic expression
                pass
        # do not check layout
        return True
    elif isinstance(lhs, FuncType) and isinstance(rhs, FuncType):
        assert lhs.param_types is not None and lhs.ret_type is not None
        assert rhs.param_types is not None and rhs.ret_type is not None
        if len(lhs.param_types) != len(rhs.param_types):
            return False
        if not type_equal(lhs.ret_type, rhs.ret_type):
            return False
        for a, b in zip(lhs.param_types, rhs.param_types):
            if not type_equal(a, b):
                return False
        return True
    elif isinstance(lhs, OpaqueType) and isinstance(rhs, OpaqueType):
        return lhs.cpp_name == rhs.cpp_name
    else:
        raise NotImplementedError()
