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
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union

import tvm_ffi
from tvm_ffi.dataclasses import field, py_class

from tilus.hidet.ir.node import Node

if TYPE_CHECKING:
    from tilus.hidet.ir.expr import Expr

Int = Union[int, "Expr"]


@py_class("tilus.hidet.ir.BaseType", frozen=True, structural_eq="tree")
class BaseType(Node):
    """Root of the hidet type hierarchy."""

    def __invert__(self) -> BaseType:
        # pointer-of: `~T` yields a pointer-type to T.
        if isinstance(self, TensorType):
            return TensorPointerType(tensor_type=self)
        if isinstance(self, DataType):
            return PointerType(base_type=self)
        if isinstance(self, (PointerType, TensorPointerType)):
            return PointerType(base_type=self)
        raise ValueError("Can not recognize type {}".format(self))

    def is_void(self) -> bool:
        return isinstance(self, VoidType)

    def is_tensor(self) -> bool:
        return isinstance(self, TensorType)

    def is_pointer(self) -> bool:
        return isinstance(self, (PointerType, TensorPointerType))

    def is_data_type(self) -> bool:
        return isinstance(self, DataType)

    def is_func_type(self) -> bool:
        return isinstance(self, FuncType)

    def is_string_type(self) -> bool:
        return isinstance(self, StringType)

    def as_data_type(self) -> Optional[DataType]:
        return self if isinstance(self, DataType) else None


@py_class("tilus.hidet.ir.DataType", frozen=True, structural_eq="tree")
class DataType(BaseType):
    """Scalar data type: its ``name`` uniquely identifies it across the module."""

    name: str
    short_name: str
    nbytes: int

    def __str__(self) -> str:
        return "hidet.{}".format(self.name)

    def __call__(self, value: Any):
        """Construct a constant, or cast an existing expression, to this dtype."""
        from tilus.hidet.ir import expr  # noqa: PLC0415

        built_types = (int, float, bool, complex)
        if (
            isinstance(value, built_types)
            or isinstance(value, (list, tuple))
            and all(isinstance(v, built_types) for v in value)
        ):
            return self.constant(value)
        if isinstance(value, expr.Constant):
            return self.constant(value.value)
        if isinstance(value, expr.Expr):
            return expr.cast(value, self)
        raise ValueError("Can not convert {} to {}".format(value, self))

    def __getitem__(self, item):
        if not isinstance(item, (tuple, list)):
            item = (item,)
        return tensor_type(dtype=self, shape=list(item))

    # Subclasses can override .nbits and .storage for subbyte dtypes.
    @property
    def nbits(self) -> int:
        return self.nbytes * 8

    @property
    def storage(self) -> DataType:
        return self

    def is_integer_subbyte(self) -> bool:
        return self.is_integer() and self.is_subbyte()

    def is_float_subbyte(self) -> bool:
        return self.is_float() and self.is_subbyte()

    def is_subbyte(self) -> bool:
        return self.nbits < 8

    def is_any_float16(self) -> bool:
        return self.is_float() and self.nbits == 16

    def is_float(self) -> bool:
        raise NotImplementedError()

    def is_integer(self) -> bool:
        raise NotImplementedError()

    def is_complex(self) -> bool:
        raise NotImplementedError()

    def is_vector(self) -> bool:
        raise NotImplementedError()

    def is_boolean(self) -> bool:
        raise NotImplementedError()

    def constant(self, value: Any):
        raise NotImplementedError()

    @property
    def one(self):
        raise NotImplementedError()

    @property
    def zero(self):
        raise NotImplementedError()

    @property
    def min_value(self):
        raise NotImplementedError()

    @property
    def max_value(self):
        raise NotImplementedError()


@py_class("tilus.hidet.ir.TensorType", frozen=True, structural_eq="tree")
class TensorType(BaseType):
    """A multi-dimensional tensor type."""

    dtype: DataType
    shape: tuple[tvm_ffi.Object, ...]  # Expr — kept as Object to avoid a circular import

    def __invert__(self) -> TensorPointerType:
        return TensorPointerType(tensor_type=self)

    @property
    def size(self) -> Expr:
        from tilus.hidet.utils import prod  # noqa: PLC0415

        return prod(list(self.shape))

    def storage_bytes(self):
        if self.dtype.is_integer_subbyte():
            return self.size * self.dtype.nbits // 8
        return self.size * self.dtype.nbytes

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]


@py_class("tilus.hidet.ir.VoidType", frozen=True, structural_eq="tree")
class VoidType(BaseType):
    pass


@py_class("tilus.hidet.ir.StringType", frozen=True, structural_eq="tree")
class StringType(BaseType):
    pass


@py_class("tilus.hidet.ir.PointerType", frozen=True, structural_eq="tree")
class PointerType(BaseType):
    """Raw pointer type; ``base_type`` is the pointee type."""

    base_type: BaseType
    specifiers: tuple[str, ...] = field(default=())
    use_bracket: bool = False

    @classmethod
    def create(cls, base_type, specifiers=(), use_bracket: bool = False) -> "PointerType":
        """Build a ``PointerType``, coercing a ``str`` base_type to a DataType."""
        if isinstance(base_type, str):
            base_type = data_type(base_type)
        return cls(
            base_type=base_type,
            specifiers=tuple(specifiers) if specifiers else (),
            use_bracket=use_bracket,
        )

    def __call__(self, x):
        from tilus.hidet.ir.expr import Constant, Expr, cast, constant  # noqa: PLC0415

        if isinstance(x, int):
            return constant(x, self)
        if isinstance(x, Constant):
            return constant(x.value, self)
        if isinstance(x, Expr):
            return cast(x, self)
        raise ValueError("Can not convert {} to {}".format(x, self))


@py_class("tilus.hidet.ir.TensorPointerType", frozen=True, structural_eq="tree")
class TensorPointerType(BaseType):
    """Pointer to a tensor of known shape/dtype."""

    tensor_type: TensorType

    @staticmethod
    def from_tensor_type(tp: TensorType) -> TensorPointerType:
        return TensorPointerType(tensor_type=tp)


@py_class("tilus.hidet.ir.FuncType", frozen=True, structural_eq="tree")
class FuncType(BaseType):
    """Function type: parameter types and return type."""

    param_types: tuple[BaseType, ...]
    ret_type: BaseType

    @staticmethod
    def from_func(func) -> FuncType:
        return FuncType(
            param_types=tuple(param.type for param in func.params),
            ret_type=func.ret_type,
        )


@py_class("tilus.hidet.ir.OpaqueType", frozen=True, structural_eq="tree")
class OpaqueType(BaseType):
    """An opaque C++ type referenced by name, e.g. ``cuda::barrier``."""

    cpp_name: str
    modifiers: tuple[str, ...] = field(default=())


# ---------------------------------------------------------------------------
# Factory helpers (preserve the pre-refactor call surface)
# ---------------------------------------------------------------------------


def tensor_type(dtype, shape: Sequence[Union[int, "Expr"]]) -> TensorType:
    from tilus.hidet.ir.expr import convert  # noqa: PLC0415
    from tilus.hidet.ir.node import is_seq  # noqa: PLC0415

    if isinstance(dtype, str):
        dtype = data_type(dtype)
    if not isinstance(dtype, DataType):
        raise ValueError('Scalar type expect a "str" or "ScalarType", but got {}'.format(type(dtype)))
    if shape is None:
        raise ValueError("Tensor type must give a shape")
    assert is_seq(shape)
    shape_tuple = tuple(convert(list(shape)))
    return TensorType(dtype=dtype, shape=shape_tuple)


def pointer_type(base_type) -> PointerType:
    return PointerType.create(base_type=base_type)


def tensor_pointer_type(dtype, shape) -> TensorPointerType:
    return TensorPointerType(tensor_type=tensor_type(dtype, shape))


def string_type() -> StringType:
    return StringType()


def func_type(param_types, ret_type) -> FuncType:
    return FuncType(param_types=tuple(param_types), ret_type=ret_type)


def data_type(dtype: Union[str, DataType]) -> DataType:
    from tilus.hidet.ir.dtypes import name2dtype, sname2dtype  # noqa: PLC0415

    if isinstance(dtype, DataType):
        return dtype
    if isinstance(dtype, str):
        if dtype in name2dtype:
            return name2dtype[dtype]
        if dtype in sname2dtype:
            return sname2dtype[dtype]
        raise ValueError("Unknown data type: {}, candidates:\n{}".format(dtype, "\n".join(name2dtype.keys())))
    raise ValueError("Expect a string or a DataType, but got {}".format(type(dtype)))


def type_equal(lhs: BaseType, rhs: BaseType) -> bool:
    """Structural type comparison that ignores symbolic tensor-shape mismatches.

    Plain ``lhs == rhs`` would require every shape element to be structurally
    identical; this helper mirrors the pre-refactor behavior that only
    compares constant shape dimensions.
    """
    if type(lhs) is not type(rhs):
        return False
    if isinstance(lhs, DataType):
        return lhs.name == rhs.name
    if isinstance(lhs, PointerType):
        return type_equal(lhs.base_type, rhs.base_type)
    if isinstance(lhs, VoidType):
        return True
    if isinstance(lhs, StringType):
        return True
    if isinstance(lhs, TensorPointerType):
        return type_equal(lhs.tensor_type, rhs.tensor_type)
    if isinstance(lhs, TensorType):
        from tilus.hidet.ir.expr import is_constant  # noqa: PLC0415

        if not type_equal(lhs.dtype, rhs.dtype):
            return False
        if len(lhs.shape) != len(rhs.shape):
            return False
        for a, b in zip(lhs.shape, rhs.shape):
            if is_constant(a) ^ is_constant(b):
                return False
            if is_constant(a) and is_constant(b) and int(a) != int(b):
                return False
        return True
    if isinstance(lhs, FuncType):
        if len(lhs.param_types) != len(rhs.param_types):
            return False
        if not type_equal(lhs.ret_type, rhs.ret_type):
            return False
        return all(type_equal(a, b) for a, b in zip(lhs.param_types, rhs.param_types))
    if isinstance(lhs, OpaqueType):
        return lhs.cpp_name == rhs.cpp_name
    raise NotImplementedError("type_equal not implemented for {} and {}".format(type(lhs), type(rhs)))


def sizeof(tp: BaseType) -> int:
    from tilus.hidet.utils import prod  # noqa: PLC0415

    if isinstance(tp, DataType):
        return tp.nbytes
    if isinstance(tp, (PointerType, TensorPointerType)):
        return 8  # 64-bit
    if isinstance(tp, TensorType):
        return sizeof(tp.dtype) * prod(list(tp.shape))
    raise NotImplementedError(type(tp))


def is_addressable(tp_or_var) -> bool:
    from tilus.hidet.ir.expr import Var  # noqa: PLC0415

    tp = tp_or_var.type if isinstance(tp_or_var, Var) else tp_or_var
    return isinstance(tp, (PointerType, TensorPointerType, TensorType))


def get_base_type(tp: BaseType) -> BaseType:
    if isinstance(tp, PointerType):
        return tp.base_type
    if isinstance(tp, TensorPointerType):
        return tp.tensor_type.dtype
    if isinstance(tp, TensorType):
        return tp.dtype
    raise AssertionError()


# module-level sentinels constructed after the classes above.
void = VoidType()
void_p = PointerType(base_type=VoidType())
