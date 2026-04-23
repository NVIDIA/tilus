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

import string
from typing import Any, Callable, ClassVar, List, Optional, Tuple, Union

from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.expr import Call, Expr, Var
from tilus.hidet.ir.node import Node
from tilus.hidet.ir.stmt import Stmt
from tilus.hidet.ir.type import BaseType


def check_func_name(name: str) -> None:
    if len(name) == 0:
        raise ValueError("Do not allow empty function name.")
    for c in name:
        if not (c in string.ascii_lowercase or c in string.ascii_uppercase or c in string.digits or c in "_"):
            raise ValueError("Cannot use {} in function name".format(repr(c)))


Dim3Like = Union[int, Expr, Tuple[Union[int, Expr], ...], List[Union[int, Expr]]]


@py_class("tilus.hidet.ir.FuncAttrs", frozen=True, structural_eq="tree")
class FuncAttrs(Node):
    """Typed attributes attached to a :class:`Function`.

    The launch-configuration fields are only meaningful for CUDA/HIP kernel
    functions. Field types stay deliberately loose (``tuple`` of ``Expr``
    values or a single scalar) so the transpiler can attach user-supplied
    ``blocks = (m, n)`` tuples without further normalization at construction
    time.
    """

    # Use the Dim3Like type implicitly by accepting Expr or int; the FFI field
    # schema accepts Python objects that are FFI-compatible. We represent
    # multi-element dims as a tuple of Expr.
    grid_dim: Any = None
    cluster_dim: Any = None
    block_dim: Any = None
    dynamic_smem_bytes: Any = None
    min_blocks: Optional[int] = None

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "grid_dim",
        "cluster_dim",
        "block_dim",
        "dynamic_smem_bytes",
        "min_blocks",
    )

    def replace(self, **kwargs: Any) -> "FuncAttrs":
        """Return a copy with the given fields replaced (py_class doesn't
        support ``dataclasses.replace`` directly)."""
        values = {name: getattr(self, name) for name in self._FIELDS}
        values.update(kwargs)
        return FuncAttrs(**values)

    def map(self, fn: Callable[[Any], Any]) -> "FuncAttrs":
        """Identity-preserving map: return self unchanged when ``fn`` is a
        no-op for every non-None field. Required by fixed-point rewriters."""
        changes: dict[str, Any] = {}
        for name in self._FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            new_value = fn(value)
            if new_value is not value:
                changes[name] = new_value
        return self.replace(**changes) if changes else self


_ALLOWED_KINDS = frozenset(
    (
        "cuda_kernel",
        "cuda_internal",
        "hip_kernel",
        "hip_internal",
        "cpu_kernel",
        "cpu_internal",
        "public",
    )
)


@py_class("tilus.hidet.ir.Function", frozen=True, structural_eq="tree")
class Function(Node):
    """Function node.

    Python ``==`` / ``hash()`` stay as the default ``tvm_ffi.Object``
    handle-identity — Functions are unique compilation entities and
    identity is what passes key their metadata on. The ``structural_eq``
    declaration is needed so that :class:`tvm_ffi.StructuralKey` can hash
    containers that hold Functions (e.g. ``IRModule.functions``).
    """

    name: str
    params: tuple[Var, ...]
    body: Stmt
    ret_type: BaseType
    kind: str
    attrs: FuncAttrs

    def __call__(self, *args, **kwargs) -> Call:
        raise ValueError("Can only call script function in another script function, or lower it to execute.")


def make_function(
    name: str,
    params: Union[List[Var], Tuple[Var, ...]],
    body: Stmt,
    ret_type: BaseType,
    kind: str,
    attrs: Optional[FuncAttrs] = None,
) -> Function:
    check_func_name(name)
    if kind not in _ALLOWED_KINDS:
        raise AssertionError(f"Invalid function kind: {kind!r}")
    return Function(
        name=name,
        params=tuple(params),
        body=body,
        ret_type=ret_type,
        kind=kind,
        attrs=attrs if attrs is not None else FuncAttrs(),
    )