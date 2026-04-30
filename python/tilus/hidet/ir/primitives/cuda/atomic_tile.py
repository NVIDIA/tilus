# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""PTX ``atom.*`` and ``red.*`` primitives for tile-level atomic instructions.

Each call lazily registers the underlying hidet primitive the first time a
given ``(op, sem, scope, space, dtype, form)`` combination is needed. This
keeps the registry footprint proportional to what the compiled program
actually uses rather than eagerly enumerating all variants.

Form:
- ``atom_rmw(...)`` → emits ``atom.{sem}.{scope}.{space}.{op}.{dtype}`` and
  returns the old value as an ``Expr``. Supports ``op`` in ``add / min / max /
  exch`` and ``op="cas"`` with an extra ``compare`` argument.
- ``red_rmw(...)`` → emits the destination-less ``red.{sem}.{scope}.{space}.
  {op}.{dtype}`` and returns a void call. Supports ``add / min / max`` only
  (PTX ``red.*`` does not define ``exch`` or ``cas``).

Address types:
- ``space="global"`` → ``addr`` is typed as ``~dtype`` (a 64-bit generic
  pointer), which hidet's asm constraint mapper translates to ``"l"``.
- ``space="shared"`` → ``addr`` is typed as ``uint32`` (the 32-bit shared-
  space integer address produced by ``cvta.to.shared.u32``), which maps to
  ``"r"``.

Supported dtypes in v1: ``int32`` and ``uint32``. Extending to ``float32`` or
``float64`` is a matter of adding entries to :data:`_PTX_DTYPE_SUFFIX`.

Note: ``from __future__ import annotations`` is intentionally NOT used in
this module because hidet's ``@script`` decorator inspects parameter type
annotations at decoration time and does not support string-form annotations.
"""

from typing import Dict, FrozenSet, Optional

from tilus.hidet.ir.dtypes import uint32
from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.primitives.func import call_primitive_func, is_primitive_function, register_primitive_function
from tilus.hidet.ir.type import DataType

# Map hidet DataType.name -> PTX dtype suffix. Note that hidet's `short_name`
# uses `i32` for signed ints whereas PTX spells it `s32`, so we keep our own
# mapping table instead of relying on `short_name`.
_PTX_DTYPE_SUFFIX: Dict[str, str] = {
    "int32": "s32",
    "uint32": "u32",
}

# Allowed PTX memory-ordering semantics / sync scopes / state spaces.
_SEMS: FrozenSet[str] = frozenset({"relaxed", "acquire", "release", "acq_rel"})
_SCOPES: FrozenSet[str] = frozenset({"cta", "cluster", "gpu", "sys"})
_SPACES: FrozenSet[str] = frozenset({"global", "shared"})

# Ops expressible by `red.*` (destination-less). `exch`/`cas` always need an
# output register, so they only ride on `atom.*`.
_RED_OPS: FrozenSet[str] = frozenset({"add", "min", "max"})


def _atom_func_name(op: str, sem: str, scope: str, space: str, dtype_name: str, form: str) -> str:
    # form is one of: "atom", "red", "cas"
    return f"cuda_{form}_{op}_{sem}_{scope}_{space}_{dtype_name}"


def _check(op: str, sem: str, scope: str, space: str, dtype: DataType) -> None:
    if sem not in _SEMS:
        raise ValueError(f"atom/red sem must be one of {sorted(_SEMS)}, got {sem!r}")
    if scope not in _SCOPES:
        raise ValueError(f"atom/red scope must be one of {sorted(_SCOPES)}, got {scope!r}")
    if space not in _SPACES:
        raise ValueError(f"atom/red space must be one of {sorted(_SPACES)}, got {space!r}")
    if dtype.name not in _PTX_DTYPE_SUFFIX:
        raise NotImplementedError(
            f"atom/red on dtype {dtype.name!r} is not supported yet. Supported: {sorted(_PTX_DTYPE_SUFFIX)}"
        )


def _register_if_needed(
    *,
    op: str,
    sem: str,
    scope: str,
    space: str,
    dtype: DataType,
    form: str,  # "atom" | "red" | "cas"
) -> str:
    """Register the primitive for this (op, sem, scope, space, dtype, form) if not already, return the name."""
    from tilus.hidet.lang import asm, attrs, script

    _check(op, sem, scope, space, dtype)
    dt_suffix = _PTX_DTYPE_SUFFIX[dtype.name]
    func_name = _atom_func_name(op, sem, scope, space, dtype.name, form)
    if is_primitive_function(func_name):
        return func_name

    # Build the PTX template.
    if form == "cas":
        template = f"atom.{sem}.{scope}.{space}.cas.{dt_suffix} %0, [%1], %2, %3;"
    elif form == "atom":
        template = f"atom.{sem}.{scope}.{space}.{op}.{dt_suffix} %0, [%1], %2;"
    elif form == "red":
        if op not in _RED_OPS:
            raise ValueError(f"red.* does not support op={op!r}; supported: {sorted(_RED_OPS)}")
        template = f"red.{sem}.{scope}.{space}.{op}.{dt_suffix} [%0], %1;"
    else:
        raise ValueError(f"unknown form {form!r}")

    # Emit the @script function. Closures over `template`, `func_name`, and
    # `dtype` are baked in at decoration time; `addr` type differs per space.
    if space == "global":
        if form == "cas":

            @script
            def func(addr: ~dtype, cmp_val: dtype, new_val: dtype) -> dtype:
                attrs.func_kind = "cuda_internal"
                attrs.func_name = func_name
                ret = dtype.zero
                asm(template, outputs=[ret], inputs=[addr, cmp_val, new_val], is_volatile=True)
                return ret

        elif form == "atom":

            @script
            def func(addr: ~dtype, v: dtype) -> dtype:
                attrs.func_kind = "cuda_internal"
                attrs.func_name = func_name
                ret = dtype.zero
                asm(template, outputs=[ret], inputs=[addr, v], is_volatile=True)
                return ret

        else:  # red

            @script
            def func(addr: ~dtype, v: dtype):
                attrs.func_kind = "cuda_internal"
                attrs.func_name = func_name
                asm(template, inputs=[addr, v], is_volatile=True)

    else:  # shared: addr is uint32 (shared-space address)
        if form == "cas":

            @script
            def func(addr: uint32, cmp_val: dtype, new_val: dtype) -> dtype:
                attrs.func_kind = "cuda_internal"
                attrs.func_name = func_name
                ret = dtype.zero
                asm(template, outputs=[ret], inputs=[addr, cmp_val, new_val], is_volatile=True)
                return ret

        elif form == "atom":

            @script
            def func(addr: uint32, v: dtype) -> dtype:
                attrs.func_kind = "cuda_internal"
                attrs.func_name = func_name
                ret = dtype.zero
                asm(template, outputs=[ret], inputs=[addr, v], is_volatile=True)
                return ret

        else:  # red

            @script
            def func(addr: uint32, v: dtype):
                attrs.func_kind = "cuda_internal"
                attrs.func_name = func_name
                asm(template, inputs=[addr, v], is_volatile=True)

    register_primitive_function(name=func_name, func_or_type=func)
    return func_name


def atom_rmw(
    *,
    op: str,
    sem: str,
    scope: str,
    space: str,
    dtype: DataType,
    addr: Expr,
    value: Expr,
    compare: Optional[Expr] = None,
) -> Expr:
    """Emit ``atom.{sem}.{scope}.{space}.{op}.{dtype}`` returning the old value.

    For ``op="cas"`` pass the compare operand; it is rejected for every other op.
    """
    if op == "cas":
        if compare is None:
            raise ValueError("atom.cas requires a compare operand")
        name = _register_if_needed(op=op, sem=sem, scope=scope, space=space, dtype=dtype, form="cas")
        return call_primitive_func(name, [addr, compare, value])
    if compare is not None:
        raise ValueError(f"compare is only valid for op='cas', got op={op!r}")
    name = _register_if_needed(op=op, sem=sem, scope=scope, space=space, dtype=dtype, form="atom")
    return call_primitive_func(name, [addr, value])


def red_rmw(
    *,
    op: str,
    sem: str,
    scope: str,
    space: str,
    dtype: DataType,
    addr: Expr,
    value: Expr,
) -> Expr:
    """Emit the destination-less ``red.{sem}.{scope}.{space}.{op}.{dtype}``.

    Supported ops: ``add``, ``min``, ``max``. Returns a void call expression the
    caller should ``append`` into the current scope.
    """
    name = _register_if_needed(op=op, sem=sem, scope=scope, space=space, dtype=dtype, form="red")
    return call_primitive_func(name, [addr, value])
