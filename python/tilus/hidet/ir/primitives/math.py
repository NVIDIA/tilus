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
from typing import Dict, Optional, Tuple

from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.tools import infer_type
from tilus.hidet.ir.type import DataType
from tilus.hidet.ir.utils.type_utils import numeric_promotion_for_all


class MathFunctionSet:
    # cast function
    def cast(self, a: Expr, cast_dtype: DataType) -> Optional[Expr]:
        """
        Cast expression a to cast_dtype.

        If the default implementation is used, return None. The default implementation is to use the cast expression
        in the underlying language (e.g. C cast):
          (cast_dtype)(a)

        Parameters
        ----------
        a: Expr
            The expression to be cast.

        cast_dtype: DataType
            The target data type.

        Returns
        -------
        ret:
            The cast expression. None if (cast_dtype)(a) is used to represent the cast.
        """
        return None

    def make_vector(self, *items: Expr) -> Expr:
        """
        Make a vector-type value from a list of sub-expressions.

        For example, if we want to create a f16x2 type given two f16 expressions, we can use the following code:
            f16x2_expr = make_vector([f16_expr1, f16_expr2])

        Parameters
        ----------
        *items: Expr
            The list of sub-expressions.

        Returns
        -------
        ret: Expr
            The vector-type value. The number of lanes is determined by the length of the list.
        """
        raise NotImplementedError()

    def make_vector_from_scalar(self, scalar: Expr, num_lanes: int) -> Expr:
        raise NotImplementedError()

    # unary math functions
    def sin(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def cos(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def tan(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def sinh(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def cosh(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def tanh(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def asin(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def acos(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def atan(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def asinh(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def acosh(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def atanh(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def exp(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def exp2(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def expm1(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def erf(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def sqrt(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def rsqrt(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def log(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def log2(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def log10(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def log1p(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def round(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def abs(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def trunc(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def ceil(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def floor(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def isfinite(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def isinf(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def isnan(self, a: Expr) -> Expr:
        raise NotImplementedError()

    # binary math functions
    def min(self, a: Expr, b: Expr) -> Expr:
        raise NotImplementedError()

    def max(self, a: Expr, b: Expr) -> Expr:
        raise NotImplementedError()

    def mod(self, a: Expr, b: Expr) -> Expr:
        raise NotImplementedError()

    def pow(self, a: Expr, b: Expr) -> Expr:
        raise NotImplementedError()

    def atan2(self, a: Expr, b: Expr) -> Expr:
        raise NotImplementedError()

    # ternary math functions
    def fma(self, a: Expr, b: Expr, c: Expr) -> Expr:
        raise NotImplementedError()


# (device, dtype) -> math function set
# such as ('cuda', 'float16') -> MathFunctionSet
registered_math_function_sets: Dict[Tuple[str, str], MathFunctionSet] = {}


def register_math_function_set(device: str, dtype: str, math_function_set: MathFunctionSet):
    if (device, dtype) in registered_math_function_sets:
        raise ValueError(f"Math function set for {device} and {dtype} already registered")
    registered_math_function_sets[(device, dtype)] = math_function_set


def _arg_dtype(a: Expr) -> DataType:
    t = infer_type(a)
    if not isinstance(t, DataType):
        raise TypeError(f"math function expects a scalar data-typed argument, got {t}")
    return t


def _lookup_set(dtype: DataType) -> MathFunctionSet:
    key = ("cuda", dtype.name)
    if key not in registered_math_function_sets:
        raise NotImplementedError(
            "No math function set registered for device 'cuda' and dtype '{}'. Registered: {}".format(
                dtype.name, list(registered_math_function_sets.keys())
            )
        )
    return registered_math_function_sets[key]


def _promote(*args: Expr) -> Tuple[DataType, Tuple[Expr, ...]]:
    from tilus.hidet.ir.expr import cast

    dtypes = tuple(_arg_dtype(a) for a in args)
    target = numeric_promotion_for_all(*dtypes)
    casted = tuple(a if d.name == target.name else cast(a, target) for a, d in zip(args, dtypes))
    return target, casted


def _unary(method_name: str, a: Expr) -> Expr:
    target, (a_c,) = _promote(a)
    return getattr(_lookup_set(target), method_name)(a_c)


def _binary(method_name: str, a: Expr, b: Expr) -> Expr:
    target, (a_c, b_c) = _promote(a, b)
    return getattr(_lookup_set(target), method_name)(a_c, b_c)


def _ternary(method_name: str, a: Expr, b: Expr, c: Expr) -> Expr:
    target, (a_c, b_c, c_c) = _promote(a, b, c)
    return getattr(_lookup_set(target), method_name)(a_c, b_c, c_c)


def _predicate(method_name: str, a: Expr) -> Expr:
    # isfinite/isinf/isnan: arg dtype preserved, return type is bool
    dtype = _arg_dtype(a)
    return getattr(_lookup_set(dtype), method_name)(a)


def sin(a: Expr) -> Expr:
    return _unary("sin", a)


def cos(a: Expr) -> Expr:
    return _unary("cos", a)


def tan(a: Expr) -> Expr:
    return _unary("tan", a)


def sinh(a: Expr) -> Expr:
    return _unary("sinh", a)


def cosh(a: Expr) -> Expr:
    return _unary("cosh", a)


def tanh(a: Expr) -> Expr:
    return _unary("tanh", a)


def asin(a: Expr) -> Expr:
    return _unary("asin", a)


def acos(a: Expr) -> Expr:
    return _unary("acos", a)


def atan(a: Expr) -> Expr:
    return _unary("atan", a)


def atan2(a: Expr, b: Expr) -> Expr:
    return _binary("atan2", a, b)


def asinh(a: Expr) -> Expr:
    return _unary("asinh", a)


def acosh(a: Expr) -> Expr:
    return _unary("acosh", a)


def atanh(a: Expr) -> Expr:
    return _unary("atanh", a)


def exp(a: Expr) -> Expr:
    return _unary("exp", a)


def exp2(a: Expr) -> Expr:
    return _unary("exp2", a)


def expm1(a: Expr) -> Expr:
    return _unary("expm1", a)


def erf(a: Expr) -> Expr:
    return _unary("erf", a)


def sqrt(a: Expr) -> Expr:
    return _unary("sqrt", a)


def rsqrt(a: Expr) -> Expr:
    return _unary("rsqrt", a)


def log(a: Expr) -> Expr:
    return _unary("log", a)


def log2(a: Expr) -> Expr:
    return _unary("log2", a)


def log10(a: Expr) -> Expr:
    return _unary("log10", a)


def log1p(a: Expr) -> Expr:
    return _unary("log1p", a)


def round(a: Expr) -> Expr:
    return _unary("round", a)


def abs(a: Expr) -> Expr:
    return _unary("abs", a)


def ceil(a: Expr) -> Expr:
    return _unary("ceil", a)


def floor(a: Expr) -> Expr:
    return _unary("floor", a)


def trunc(a: Expr) -> Expr:
    return _unary("trunc", a)


def min(a: Expr, b: Expr) -> Expr:
    return _binary("min", a, b)


def max(a: Expr, b: Expr) -> Expr:
    return _binary("max", a, b)


def mod(a: Expr, b: Expr) -> Expr:
    return _binary("mod", a, b)


def pow(a: Expr, b: Expr) -> Expr:
    return _binary("pow", a, b)


def fma(a: Expr, b: Expr, c: Expr) -> Expr:
    return _ternary("fma", a, b, c)


def isfinite(a: Expr) -> Expr:
    return _predicate("isfinite", a)


def isinf(a: Expr) -> Expr:
    return _predicate("isinf", a)


def isnan(a: Expr) -> Expr:
    return _predicate("isnan", a)


def make_vector(*args: Expr) -> Expr:
    if len(args) == 0:
        raise ValueError("make_vector requires at least one argument")
    dtypes = [_arg_dtype(a) for a in args]
    if not all(d.name == dtypes[0].name for d in dtypes[1:]):
        raise ValueError("make_vector requires all arguments to have the same dtype, got {}".format(dtypes))
    return _lookup_set(dtypes[0]).make_vector(*args)
