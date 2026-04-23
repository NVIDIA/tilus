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
# pylint: disable=redefined-builtin
from __future__ import annotations

import operator
import string
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.dtypes import boolean, promote_type

from .dtypes import default_float_dtype, default_int_dtype
from .node import Node
from .type import (
    BaseType,
    DataType,
    FuncType,
    PointerType,
    StringType,
    TensorPointerType,
    TensorType,
    data_type,
    string_type,
    tensor_pointer_type,
    tensor_type,
)

PyScalar = Union[bool, int, float, complex, str]


# ---------------------------------------------------------------------------
# Base expression and arithmetic-operator overloads
# ---------------------------------------------------------------------------


@py_class("tilus.hidet.ir.Expr", frozen=True, structural_eq="tree")
class Expr(Node):
    """Base class for every hidet IR expression node.

    Python ``==`` and ``hash()`` stay as the default handle-identity from
    ``tvm_ffi.Object`` — two distinct instances never compare equal, and a
    plain ``dict`` keyed on an Expr acts as an identity map even across the
    fresh Python wrappers that FFI-container indexing returns.

    Each subclass declares ``structural_eq="tree"`` so that
    :func:`tvm_ffi.structural_equal` / :func:`tvm_ffi.structural_hash` walk
    its fields when needed. Use :class:`~tilus.hidet.ir.node.NodeDict` /
    :class:`~tilus.hidet.ir.node.NodeSet` (backed by
    :class:`tvm_ffi.StructuralKey`) to key a container by structural
    equality.

    Python comparison operators are routed through the
    :func:`equal` / :func:`not_equal` / :func:`less_than` / :func:`less_equal`
    factory helpers and the ordering dunders below; they all build IR nodes
    rather than returning Python bools.
    """

    def __bool__(self) -> bool:
        raise TypeError(
            "hidet.ir.Expr does not support pythonic logical operations (e.g., and, or, not, if(...)). "
            "Please use hidet.ir.if_then_else, hidet.ir.logical_and, hidet.ir.logical_or, hidet.ir.logical_or "
            "explicitly."
        )

    def __call__(self, *args, **kwargs):
        if len(kwargs) > 0:
            raise ValueError("Keyword arguments are not supported in hidet function calls.")
        if isinstance(self, Var) and self.type.is_func_type():
            return call(self, args)
        raise ValueError("Only function variable can be called.")

    def __neg__(self):
        return _unary(Neg, self)

    def __add__(self, other):
        return _binary(Add, self, other)

    def __radd__(self, other):
        return _binary(Add, other, self)

    def __sub__(self, other):
        return _binary(Sub, self, other)

    def __rsub__(self, other):
        return _binary(Sub, other, self)

    def __mul__(self, other):
        return _binary(Multiply, self, other)

    def __rmul__(self, other):
        return _binary(Multiply, other, self)

    def __truediv__(self, other):
        return _binary(Div, self, other)

    def __rtruediv__(self, other):
        return _binary(Div, other, self)

    def __floordiv__(self, other):
        return _binary(Div, self, other)

    def __rfloordiv__(self, other):
        return _binary(Div, other, self)

    def __mod__(self, other):
        return _binary(Mod, self, other)

    def __rmod__(self, other):
        return _binary(Mod, other, self)

    def __lshift__(self, other):
        return _binary(LeftShift, self, other)

    def __rshift__(self, other):
        return _binary(RightShift, self, other)

    def __rlshift__(self, other):
        return _binary(LeftShift, other, self)

    def __rrshift__(self, other):
        return _binary(RightShift, other, self)

    # Ordering operators stay as IR-construction helpers. py_class does NOT
    # auto-generate ``__lt__/__le__/__gt__/__ge__`` unless ``order=True``,
    # so they're safe to use for IR construction. Equality / inequality
    # (``==`` / ``!=``) keep the Object-default handle-identity and are
    # reserved for Python dict/set keying; use ``equal(a, b)`` /
    # ``not_equal(a, b)`` when you want to build an IR comparison node.
    def __lt__(self, other):
        return _binary(LessThan, self, other)

    def __le__(self, other):
        return _binary(LessEqual, self, other)

    def __gt__(self, other):
        return _binary(LessThan, other, self)

    def __ge__(self, other):
        return _binary(LessEqual, other, self)

    def __invert__(self):
        # ~a is overloaded as the C "take address of a" operator.
        return Address(expr=self)

    def __or__(self, other):
        return _binary(BitwiseOr, self, other)

    def __ror__(self, other):
        return _binary(BitwiseOr, other, self)

    def __and__(self, other):
        return _binary(BitwiseAnd, self, other)

    def __rand__(self, other):
        return _binary(BitwiseAnd, other, self)

    def __xor__(self, other):
        return _binary(BitwiseXor, self, other)

    def __rxor__(self, other):
        return _binary(BitwiseXor, other, self)

    def __getitem__(self, items):
        if not isinstance(items, (tuple, list)):
            items = [items]
        for item in items:
            if isinstance(item, slice):
                raise ValueError("Tensor slicing is not supported in hidet IR; use explicit indices.")
        return tensor_element(base=self, indices=list(items))

    def __setitem__(self, key, value):
        raise ValueError()

    def __int__(self):
        raise TypeError("Cannot convert hidet.ir.Expr to int.")

    def __float__(self):
        raise TypeError("Cannot convert hidet.ir.Expr to float.")

    def __complex__(self):
        raise TypeError("Cannot convert hidet.ir.Expr to complex.")

    def write(self, items, value, protected=True):
        from tilus.hidet.ir.stmt import BufferStoreStmt  # noqa: PLC0415

        te = self[items]
        if not isinstance(te, TensorElement):
            raise ValueError("expect element indexing, but got slicing.")
        return BufferStoreStmt.create(buf=self, indices=te.indices, value=value, protected=protected)


# ---------------------------------------------------------------------------
# Binary / unary shape classes
# ---------------------------------------------------------------------------


@py_class("tilus.hidet.ir.BinaryExpr", frozen=True, structural_eq="tree")
class BinaryExpr(Expr):
    a: Expr
    b: Expr


@py_class("tilus.hidet.ir.UnaryExpr", frozen=True, structural_eq="tree")
class UnaryExpr(Expr):
    a: Expr


@py_class("tilus.hidet.ir.Condition", frozen=True, structural_eq="tree")
class Condition(Expr):
    """Marker for boolean-valued expressions. No fields of its own."""


@py_class("tilus.hidet.ir.LessThan", frozen=True, structural_eq="tree")
class LessThan(BinaryExpr):
    pass


@py_class("tilus.hidet.ir.LessEqual", frozen=True, structural_eq="tree")
class LessEqual(BinaryExpr):
    pass


@py_class("tilus.hidet.ir.Equal", frozen=True, structural_eq="tree")
class Equal(BinaryExpr):
    pass


@py_class("tilus.hidet.ir.NotEqual", frozen=True, structural_eq="tree")
class NotEqual(BinaryExpr):
    pass


@py_class("tilus.hidet.ir.LogicalAnd", frozen=True, structural_eq="tree")
class LogicalAnd(BinaryExpr):
    pass


@py_class("tilus.hidet.ir.LogicalOr", frozen=True, structural_eq="tree")
class LogicalOr(BinaryExpr):
    pass


@py_class("tilus.hidet.ir.LogicalNot", frozen=True, structural_eq="tree")
class LogicalNot(UnaryExpr):
    pass


@py_class("tilus.hidet.ir.Neg", frozen=True, structural_eq="tree")
class Neg(UnaryExpr):
    pass


@py_class("tilus.hidet.ir.Add", frozen=True, structural_eq="tree")
class Add(BinaryExpr):
    pass


@py_class("tilus.hidet.ir.Sub", frozen=True, structural_eq="tree")
class Sub(BinaryExpr):
    pass


@py_class("tilus.hidet.ir.Multiply", frozen=True, structural_eq="tree")
class Multiply(BinaryExpr):
    pass


@py_class("tilus.hidet.ir.Div", frozen=True, structural_eq="tree")
class Div(BinaryExpr):
    pass


@py_class("tilus.hidet.ir.Mod", frozen=True, structural_eq="tree")
class Mod(BinaryExpr):
    pass


@py_class("tilus.hidet.ir.BitwiseNot", frozen=True, structural_eq="tree")
class BitwiseNot(UnaryExpr):
    pass


@py_class("tilus.hidet.ir.BitwiseAnd", frozen=True, structural_eq="tree")
class BitwiseAnd(BinaryExpr):
    pass


@py_class("tilus.hidet.ir.BitwiseOr", frozen=True, structural_eq="tree")
class BitwiseOr(BinaryExpr):
    pass


@py_class("tilus.hidet.ir.BitwiseXor", frozen=True, structural_eq="tree")
class BitwiseXor(BinaryExpr):
    pass


@py_class("tilus.hidet.ir.LeftShift", frozen=True, structural_eq="tree")
class LeftShift(BinaryExpr):
    pass


@py_class("tilus.hidet.ir.RightShift", frozen=True, structural_eq="tree")
class RightShift(BinaryExpr):
    pass


# ---------------------------------------------------------------------------
# Compound expressions
# ---------------------------------------------------------------------------


@py_class("tilus.hidet.ir.TensorElement", frozen=True, structural_eq="tree")
class TensorElement(Expr):
    base: Expr
    indices: tuple[Expr, ...]
    protected: bool = False


@py_class("tilus.hidet.ir.Call", frozen=True, structural_eq="tree")
class Call(Expr):
    func_var: "Var"
    args: tuple[Expr, ...]


@py_class("tilus.hidet.ir.Let", frozen=True, structural_eq="tree")
class Let(Expr):
    var: "Var"
    value: Expr
    body: Expr


@py_class("tilus.hidet.ir.Cast", frozen=True, structural_eq="tree")
class Cast(Expr):
    expr: Expr
    target_type: BaseType


@py_class("tilus.hidet.ir.IfThenElse", frozen=True, structural_eq="tree")
class IfThenElse(Expr):
    cond: Expr
    then_expr: Expr
    else_expr: Expr


@py_class("tilus.hidet.ir.Dereference", frozen=True, structural_eq="tree")
class Dereference(Expr):
    expr: Expr


@py_class("tilus.hidet.ir.Address", frozen=True, structural_eq="tree")
class Address(Expr):
    expr: Expr


@py_class("tilus.hidet.ir.Constant", frozen=True, structural_eq="tree")
class Constant(Expr):
    """A compile-time scalar / string / pointer constant.

    ``complex`` constants are not supported: complex scalar types are
    registered but never reach CUDA codegen.
    """

    value: Union[bool, int, float, str]
    type: BaseType

    def is_scalar(self) -> bool:
        return isinstance(self.type, DataType)

    def is_string(self) -> bool:
        return isinstance(self.type, StringType)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __bool__(self):
        return bool(self.value)

    def __complex__(self):
        return complex(self.value)


# ---------------------------------------------------------------------------
# Var — identity-based, does NOT structurally compare equal to another Var
# with the same name/type. This is what hidet historically relied on.
# ---------------------------------------------------------------------------


@py_class("tilus.hidet.ir.Var", frozen=True, structural_eq="var")
class Var(Expr):
    """A named variable.

    Unlike compound expressions, ``Var`` stays on the default ``Object``
    identity semantics (``eq=False``): two freshly allocated ``Var("x",
    int32)`` instances are NOT equal. Identity is what lets SSA passes keep
    distinct definitions distinct.
    """

    name: Optional[str]
    type: BaseType


# ---------------------------------------------------------------------------
# Constant interning pool (lives outside the class because frozen py_class
# can't hold class-level mutable state).
# ---------------------------------------------------------------------------

_constant_pool: Dict[Tuple[Any, str], Constant] = {}


Int = Union[int, Expr]

operator_dict: Dict[Type[Expr], Callable] = {
    Neg: operator.neg,
    LogicalNot: operator.not_,
    BitwiseNot: operator.invert,
    Add: operator.add,
    Sub: operator.sub,
    Multiply: operator.mul,
    Div: lambda a, b: a // b if isinstance(a, int) and isinstance(b, int) else a / b,
    Mod: operator.mod,
    BitwiseOr: operator.or_,
    BitwiseAnd: operator.and_,
    BitwiseXor: operator.xor,
    Equal: operator.eq,
    NotEqual: operator.ne,
    LessThan: operator.lt,
    LessEqual: operator.le,
    LogicalAnd: lambda a, b: a and b,
    LogicalOr: lambda a, b: a or b,
    LeftShift: operator.lshift,
    RightShift: operator.rshift,
}


# ---------------------------------------------------------------------------
# Internal constant-folding helpers (used by operator overloads above)
# ---------------------------------------------------------------------------


def _unary(cls, a):
    if not isinstance(a, Expr):
        a = convert(a)
    if isinstance(a, Constant):
        if cls is Neg:
            return constant(-a.value, a.type)
        if cls is LogicalNot:
            return constant(not a.value, a.type)
        if cls is BitwiseNot:
            return constant(~a.value, a.type)
        raise ValueError("unknown unary operator {}".format(cls))
    return cls(a=a)


def _binary(cls, a: Expr, b: Expr):
    if not isinstance(a, Expr):
        a = convert(a)
    if not isinstance(b, Expr):
        b = convert(b)
    if isinstance(a, Constant) and isinstance(b, Constant):
        if a.type.is_data_type() and b.type.is_data_type():
            value = operator_dict[cls](a.value, b.value)
            if cls in [Equal, NotEqual, LessThan, LessEqual, LogicalAnd, LogicalOr]:
                return constant(value, const_type="bool")
            if cls in [LeftShift, RightShift]:
                return constant(value, a.type)
            return constant(value, promote_type(a.type, b.type))
        if a.type.is_pointer() and b.type.is_pointer():
            if cls is Sub:
                return constant(a.value - b.value, "uint64")
            if cls in [Equal, NotEqual]:
                return constant(operator_dict[cls](a.value, b.value), "bool")
            raise ValueError("unknown binary operator {}".format(cls))
        if a.type.is_pointer() and b.type.is_data_type():
            return constant(a.value + b.value, a.type)
        if a.type.is_data_type() and b.type.is_pointer():
            return constant(a.value + b.value, b.type)
        if a.type.is_string_type() and b.type.is_string_type():
            if cls is Add:
                return constant(a.value + b.value, a.type)
            if cls in [Equal, NotEqual]:
                return constant(operator_dict[cls](a.value, b.value), "bool")
            raise ValueError("unknown binary operator {}".format(cls))
        raise ValueError("unknown binary operator {}".format(cls))
    if isinstance(b, Constant) and b.type.is_data_type():
        from tilus.hidet.ir.tools import infer_type  # noqa: PLC0415

        if int_equal(b, 0):
            if cls in [Add, Sub, BitwiseXor, BitwiseOr]:
                return a
            if cls is Multiply:
                return promote_type(infer_type(a), b.type)(0)
        elif int_equal(b, 1) and cls in [Multiply, Div]:
            return a
    elif isinstance(a, Constant):
        from tilus.hidet.ir.tools import infer_type  # noqa: PLC0415

        if int_equal(a, 0):
            if cls in [Add, BitwiseXor, BitwiseOr]:
                return b
            if cls is Multiply:
                return promote_type(infer_type(b), a.type)(0)
        elif int_equal(a, 1) and cls is Multiply:
            return b
    elif a.same_as(b) and isinstance(a, Var) and cls is LessThan:
        return constant(False, "bool")

    return cls(a=a, b=b)


def int_equal(c: Expr, value: int) -> bool:
    """True iff ``c`` is a numeric Constant whose Python value equals ``value``."""
    return isinstance(c, Constant) and isinstance(c.value, (int, float)) and c.value == value


# ---------------------------------------------------------------------------
# Public factory helpers / comparison builders
# ---------------------------------------------------------------------------


def convert(
    obj: Optional[Union[Expr, PyScalar, tuple, Sequence]],
    dtype: Optional[Union[str, DataType]] = None,
) -> Optional[Union[Expr, tuple]]:
    if isinstance(obj, Expr):
        return obj
    if dtype is not None:
        if isinstance(obj, (bool, int, float)):
            return constant(obj, dtype)
        raise ValueError("Can not convert {} to {}.".format(obj, dtype))
    if isinstance(obj, bool):
        return constant(obj, data_type("bool"))
    if isinstance(obj, int):
        return constant(obj, data_type("int32"))
    if isinstance(obj, float):
        return constant(obj, data_type("float32"))
    if isinstance(obj, str):
        return constant(obj, string_type())
    if isinstance(obj, (tuple, list)):
        return tuple(convert(v) for v in obj)
    if obj is None:
        return None
    raise NotImplementedError(type(obj))


def as_expr(obj: Union[float, bool, int, str, Expr]) -> Expr:
    if isinstance(obj, Expr):
        return obj
    if isinstance(obj, bool):
        return boolean.constant(obj)
    if isinstance(obj, int):
        assert default_int_dtype.imin <= obj <= default_int_dtype.imax, obj
        return default_int_dtype.constant(obj)
    if isinstance(obj, float):
        return default_float_dtype.constant(obj)
    if isinstance(obj, str):
        return Constant(value=obj, type=string_type())
    raise ValueError(obj)


def var(name: str = None, dtype="int32") -> Var:
    if isinstance(name, str):
        assert set(name) <= set(string.ascii_letters + "_." + string.digits)
    if isinstance(dtype, str):
        dtype = data_type(dtype)
    return Var(name=name, type=dtype)


def scalar_var(name: str, dtype: Union[str, DataType] = "float32") -> Var:
    dtype = dtype if isinstance(dtype, DataType) else data_type(dtype)
    return Var(name=name, type=dtype)


def tensor_var(name: str, shape, dtype: Union[str, DataType] = "float32") -> Var:
    return Var(name=name, type=tensor_type(dtype, shape))


def is_one(v: Expr) -> bool:
    return isinstance(v, Constant) and v.value == 1


def is_zero(v: Expr) -> bool:
    return isinstance(v, Constant) and v.value == 0


def is_true(v: Union[Expr, bool]) -> bool:
    if isinstance(v, (Constant, bool)):
        return bool(v) is True
    return False


def is_false(v: Union[Expr, bool]) -> bool:
    if isinstance(v, (Constant, bool)):
        return bool(v) is False
    return False


def if_then_else(
    cond: Union[Expr, PyScalar], then_expr: Union[Expr, PyScalar], else_expr: Union[Expr, PyScalar]
) -> Expr:
    cond = convert(cond)
    then_expr = convert(then_expr)
    else_expr = convert(else_expr)
    if is_constant(cond):
        return then_expr if bool(cond) else else_expr
    return IfThenElse(cond=cond, then_expr=then_expr, else_expr=else_expr)


def tensor_element(base: Expr, indices: Sequence[Int], protected: bool = False) -> TensorElement:
    return TensorElement(base=base, indices=tuple(convert(i) for i in indices), protected=protected)


def _chain_binary_op(op: Type[BinaryExpr], operands, default):
    if len(operands) == 0:
        return convert(default)
    if len(operands) == 1:
        return convert(operands[0])
    a = _chain_binary_op(op, operands[:-1], default)
    b = convert(operands[-1])
    return _binary(op, a, b)


def logical_and(*args: Union[Expr, bool]):
    return _chain_binary_op(LogicalAnd, args, True)


def logical_or(*args: Union[Expr, bool]):
    return _chain_binary_op(LogicalOr, args, False)


def logical_not(a: Union[Expr, PyScalar]):
    return _unary(LogicalNot, convert(a))


def bitwise_not(a: Union[Expr, PyScalar]):
    return _unary(BitwiseNot, convert(a))


def equal(a: Union[Expr, PyScalar], b: Union[Expr, PyScalar]) -> Expr:
    """Build an ``Equal(a, b)`` IR node (the structural-eq replacement
    for the old ``Expr.__eq__`` operator overload)."""
    return _binary(Equal, convert(a), convert(b))


def less_than(a: Union[Expr, PyScalar], b: Union[Expr, PyScalar]) -> Expr:
    return _binary(LessThan, convert(a), convert(b))


def less_equal(a: Union[Expr, PyScalar], b: Union[Expr, PyScalar]) -> Expr:
    return _binary(LessEqual, convert(a), convert(b))


def not_equal(a: Union[Expr, PyScalar], b: Union[Expr, PyScalar]) -> Expr:
    return _binary(NotEqual, convert(a), convert(b))


def left_shift(a: Union[Expr, int], b: Union[Expr, int]):
    return _binary(LeftShift, convert(a), convert(b))


def right_shift(a: Union[Expr, int], b: Union[Expr, int]):
    return _binary(RightShift, convert(a), convert(b))


def bitwise_and(*args: Union[Expr, int]):
    return _chain_binary_op(BitwiseAnd, args, -1)


def bitwise_or(*args: Union[Expr, int]):
    return _chain_binary_op(BitwiseOr, args, 0)


def bitwise_xor(*args: Union[Expr, int]):
    return _chain_binary_op(BitwiseXor, args, 0)


def cast(v: Union[Expr, int, bool, float], dtype: Union[str, DataType, BaseType]) -> Expr:
    if isinstance(v, (bool, int, float)):
        v = convert(v)
    if not isinstance(v, Expr):
        raise ValueError("Expect an expression, got {}".format(type(v).__name__))
    if isinstance(dtype, str):
        dtype = data_type(dtype)
    if isinstance(v, Constant) and v.is_scalar():
        if dtype.is_vector():
            raise ValueError("Can not cast a scalar {} to a vector type {}.".format(v, dtype))
        return constant(v.value, dtype)
    if isinstance(v, Var) and v.type.is_data_type() and dtype.is_data_type() and v.type == dtype:
        return v
    return Cast(expr=v, target_type=dtype)


def call(func: Var, args: Sequence[Union[Expr, PyScalar]]) -> Call:
    return Call(func_var=func, args=tuple(convert(a) for a in args))


def tensor_pointer_var(hint: str, shape, dtype: Union[str, DataType] = "float32") -> Var:
    return Var(name=hint, type=tensor_pointer_type(dtype=dtype, shape=shape))


def view(ptr: Expr, tp: TensorType) -> Expr:
    if not isinstance(tp, TensorType):
        raise ValueError("Expect a tensor type, got {}".format(type(tp).__name__))
    return cast(ptr, TensorPointerType.from_tensor_type(tp))


def address(v: Expr) -> Expr:
    return Address(expr=v)


def deref(v: Expr, derefed_type: Optional[BaseType] = None) -> Expr:
    if derefed_type is not None:
        v = cast(v, ~derefed_type)
    return Dereference(expr=v)


def is_constant(e: Union[Expr, PyScalar], *other: Union[Expr, PyScalar]) -> bool:
    if isinstance(e, Expr) and not isinstance(e, Constant):
        return False
    if len(other) > 0:
        return is_constant(*other)
    return True


def constant(value, const_type: Union[str, BaseType]) -> Constant:
    if const_type and isinstance(const_type, str):
        const_type = data_type(const_type)
    # normalize the value based on const_type
    if isinstance(const_type, DataType):
        if const_type.is_complex():
            raise NotImplementedError(
                "Complex constants are not supported in tilus.hidet.ir; "
                "the complex dtypes are registered but never reach CUDA codegen."
            )
        if const_type.is_float():
            value = float(value)
        elif const_type.is_integer():
            if const_type == boolean:
                value = bool(value)
            else:
                value = int(value)
        elif const_type.is_vector():
            # Vector constants store their lanes as a tuple of Python scalars.
            value = tuple(value)
        else:
            raise ValueError(f"Invalid data const_type {const_type}")
    elif isinstance(const_type, PointerType):
        value = int(value)
    elif isinstance(const_type, StringType):
        value = str(value)
    else:
        raise ValueError(f"Invalid const_type {const_type}")

    if const_type.is_data_type() and (
        (isinstance(value, int) and -128 <= value <= 128) or (isinstance(value, float) and value in [-1.0, 0.0, 1.0])
    ):
        key = (value, const_type.name)
        pooled = _constant_pool.get(key)
        if pooled is None:
            pooled = Constant(value=value, type=const_type)
            _constant_pool[key] = pooled
        return pooled
    return Constant(value=value, type=const_type)


def constant_int(value: int, const_type: "IntegerType") -> Constant:
    if -128 <= value <= 128:
        key = (value, const_type.name)
        pooled = _constant_pool.get(key)
        if pooled is None:
            pooled = Constant(value=value, type=const_type)
            _constant_pool[key] = pooled
        return pooled
    return Constant(value=value, type=const_type)


def index_vars(num_vars: int) -> List[Var]:
    default_names = ["i", "j", "k", "p", "q", "r", "s", "t", "u", "v"]
    if num_vars < len(default_names):
        return [var(default_names[i]) for i in range(num_vars)]
    return [var("i") for _ in range(num_vars)]


def reinterpret(value: Expr, target_type: BaseType) -> Expr:
    return cast(~value, ~target_type)[0]


# Re-export for pre-refactor call sites that depended on Expr._binary / ._unary.
setattr(Expr, "_binary", staticmethod(_binary))
setattr(Expr, "_unary", staticmethod(_unary))
