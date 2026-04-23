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

from enum import StrEnum
from typing import Any, ClassVar, List, Optional, Sequence, Tuple, Union

from tvm_ffi import Array
from tvm_ffi.dataclasses import py_class

from tilus.hidet.ir.expr import Constant, Expr, Var, convert
from tilus.hidet.ir.node import Node
from tilus.hidet.ir.type import DataType, PointerType, TensorPointerType


class DeclareScope(StrEnum):
    """Storage scope of a declared tensor variable.

    Stored as a Python ``str`` on ``@py_class`` fields. ``DeclareScope.Global``
    compares equal to the string ``"global"``, so consumer code keeps working.
    """

    Default = "default"
    Global = "global"
    Shared = "shared"
    Register = "register"
    Host = "host"

    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"

    @staticmethod
    def from_str(name: str) -> "DeclareScope":
        return {
            "global": DeclareScope.Global,
            "shared": DeclareScope.Shared,
            "register": DeclareScope.Register,
        }.get(name, DeclareScope.Default)

    def is_global(self) -> bool:
        return self is DeclareScope.Global

    def is_shared(self) -> bool:
        return self is DeclareScope.Shared

    def is_register(self) -> bool:
        return self in (DeclareScope.Register, DeclareScope.Default)

    def is_memory(self) -> bool:
        return not self.is_register()


def _scope_to_str(scope: Optional[Union[str, DeclareScope]]) -> str:
    if scope is None:
        return DeclareScope.Default.value
    if isinstance(scope, DeclareScope):
        return scope.value
    return str(scope)


@py_class("tilus.hidet.ir.ForStmtAttr", frozen=True, structural_eq="tree")
class ForStmtAttr(Node):
    unroll: bool = False
    unroll_factor: Optional[int] = None
    unroll_explicit: bool = False
    parallel: bool = False
    parallel_threads: Optional[int] = None

    def __str__(self) -> str:
        if self.unroll:
            if self.unroll_explicit:
                return "u+"
            if self.unroll_factor:
                return f"u{self.unroll_factor}"
            return "u"
        if self.parallel:
            if self.parallel_threads:
                return f"p{self.parallel_threads}"
            return "p"
        return "."

    @staticmethod
    def from_extent(extent: Union[int, Expr]) -> "ForStmtAttr":
        if isinstance(extent, Expr):
            if isinstance(extent, Constant):
                extent = int(extent)
            else:
                return ForStmtAttr()
        if extent < 4:
            return ForStmtAttr(unroll=True, unroll_explicit=True)
        return ForStmtAttr()

    @staticmethod
    def parse(attr: Optional[str], num_loops: int) -> List["ForStmtAttr"]:
        if attr is None:
            attr = ""
        s = attr.replace(" ", "")
        idx = 0

        def cur() -> Optional[str]:
            return s[idx] if idx < len(s) else None

        attrs: List[ForStmtAttr] = []
        while idx < len(s):
            if s[idx] == ".":
                idx += 1
                attrs.append(ForStmtAttr())
            elif s[idx] == "u":
                idx += 1
                c = cur()
                if c == "+":
                    attrs.append(ForStmtAttr(unroll=True, unroll_explicit=True))
                    idx += 1
                elif c and c.isdigit():
                    unroll_factor = 0
                    while c and c.isdigit():
                        unroll_factor = unroll_factor * 10 + int(c)
                        idx += 1
                        c = cur()
                    if unroll_factor == 0:
                        raise ValueError(f"Invalid attribute string: {attr}")
                    attrs.append(ForStmtAttr(unroll=True, unroll_factor=unroll_factor))
                else:
                    attrs.append(ForStmtAttr(unroll=True, unroll_explicit=False))
            elif s[idx] == "p":
                idx += 1
                c = cur()
                if c and c.isdigit():
                    parallel_threads = 0
                    while c and c.isdigit():
                        parallel_threads = parallel_threads * 10 + int(c)
                        idx += 1
                        c = cur()
                    if parallel_threads == 0:
                        raise ValueError(f"Invalid attribute string: {attr}")
                    attrs.append(ForStmtAttr(parallel=True, parallel_threads=parallel_threads))
                else:
                    attrs.append(ForStmtAttr(parallel=True))
            else:
                raise ValueError(f"Invalid attribute string: {attr}")
        if len(attrs) == 0:
            attrs = [ForStmtAttr() for _ in range(num_loops)]
        elif len(attrs) == 1:
            attrs = attrs * num_loops
        elif len(attrs) != num_loops:
            raise ValueError("Invalid attribute string: {} for {} loops".format(attr, num_loops))
        return attrs


# ---------------------------------------------------------------------------
# Statement classes.
#
# No user-defined ``__init__`` — the py_class decorator auto-generates one
# that enforces field types at the FFI boundary. Each class exposes a
# ``create(cls, ...)`` classmethod whose arguments are *strictly typed*
# (``Expr``, ``Var``, ``Stmt``, ``Sequence[Stmt]``, ...) — no ``convert()``
# coercion happens inside ``create``. Callers that start from raw Python
# values must ``convert()`` them first (the stmt_builder / transpiler /
# other IR-constructing layers own that coercion).
#
# The job of ``create`` is limited to:
#   - container materialization: ``tuple(seq)`` for ``Array[T]`` fields
#     (generators and custom Sequence objects aren't auto-accepted by
#     the FFI type converter even though ``list`` / ``tuple`` are);
#   - structural / invariant assertions (length matches, body not None,
#     target is supported, ...);
#   - enum / scope coercion (``DeclareScope`` → ``str``).
#
# Sequence-of-IR fields use ``Array[T]`` rather than ``tuple[T, ...]`` so
# the schema honestly reflects the runtime type.
# ---------------------------------------------------------------------------


@py_class("tilus.hidet.ir.Stmt", frozen=True, structural_eq="tree")
class Stmt(Node):
    pass


@py_class("tilus.hidet.ir.EvaluateStmt", frozen=True, structural_eq="tree")
class EvaluateStmt(Stmt):
    expr: Expr

    @classmethod
    def create(cls, expr: Expr) -> EvaluateStmt:
        return cls(expr=expr)


@py_class("tilus.hidet.ir.DeclareStmt", frozen=True, structural_eq="tree")
class DeclareStmt(Stmt):
    var: Var
    init: Optional[Expr] = None
    is_static: bool = False
    scope: str = DeclareScope.Default.value

    @classmethod
    def create(
        cls,
        var: Var,
        init: Optional[Expr] = None,
        is_static: bool = False,
        scope: Optional[Union[DeclareScope, str]] = None,
    ) -> DeclareStmt:
        return cls(var=var, init=init, is_static=is_static, scope=_scope_to_str(scope))


@py_class("tilus.hidet.ir.BufferStoreStmt", frozen=True, structural_eq="tree")
class BufferStoreStmt(Stmt):
    buf: Expr  # Var or TensorNode
    indices: Array[Expr]
    value: Expr
    protected: bool = False

    @classmethod
    def create(
        cls, buf: Expr, indices: Sequence[Expr], value: Expr, protected: bool = False
    ) -> BufferStoreStmt:
        return cls(buf=buf, indices=tuple(indices), value=value, protected=protected)


@py_class("tilus.hidet.ir.AssignStmt", frozen=True, structural_eq="tree")
class AssignStmt(Stmt):
    var: Var
    value: Expr

    @classmethod
    def create(cls, var: Var, value: Expr) -> AssignStmt:
        return cls(var=var, value=value)


@py_class("tilus.hidet.ir.ReturnStmt", frozen=True, structural_eq="tree")
class ReturnStmt(Stmt):
    ret_value: Optional[Expr] = None


@py_class("tilus.hidet.ir.LetStmt", frozen=True, structural_eq="tree")
class LetStmt(Stmt):
    bind_vars: Array[Var]
    bind_values: Array[Expr]
    body: Stmt

    @classmethod
    def create(cls, bind_vars: Sequence[Var], bind_values: Sequence[Expr], body: Stmt) -> LetStmt:
        bind_vars = tuple(bind_vars)
        bind_values = tuple(bind_values)
        assert len(bind_vars) == len(bind_values) > 0
        return cls(bind_vars=bind_vars, bind_values=bind_values, body=body)


@py_class("tilus.hidet.ir.ForStmt", frozen=True, structural_eq="tree")
class ForStmt(Stmt):
    DEFAULT_UNROLL_LIMIT: ClassVar[int] = 32

    loop_var: Var
    extent: Expr
    body: Stmt
    attr: ForStmtAttr

    @classmethod
    def create(
        cls, loop_var: Var, extent: Expr, body: Stmt, *, attr: Optional[ForStmtAttr] = None
    ) -> ForStmt:
        return cls(
            loop_var=loop_var,
            extent=extent,
            body=body,
            attr=attr if attr is not None else ForStmtAttr.from_extent(extent),
        )


@py_class("tilus.hidet.ir.WhileStmt", frozen=True, structural_eq="tree")
class WhileStmt(Stmt):
    cond: Expr
    body: Stmt

    @classmethod
    def create(cls, cond: Expr, body: Stmt) -> WhileStmt:
        return cls(cond=cond, body=body)


@py_class("tilus.hidet.ir.BreakStmt", frozen=True, structural_eq="tree")
class BreakStmt(Stmt):
    pass


@py_class("tilus.hidet.ir.ContinueStmt", frozen=True, structural_eq="tree")
class ContinueStmt(Stmt):
    pass


@py_class("tilus.hidet.ir.IfStmt", frozen=True, structural_eq="tree")
class IfStmt(Stmt):
    cond: Expr
    then_body: Stmt
    else_body: Optional[Stmt] = None

    @classmethod
    def create(cls, cond: Expr, then_body: Stmt, else_body: Optional[Stmt] = None) -> IfStmt:
        return cls(cond=cond, then_body=then_body, else_body=else_body)


@py_class("tilus.hidet.ir.AssertStmt", frozen=True, structural_eq="tree")
class AssertStmt(Stmt):
    cond: Expr
    msg: Optional[str] = None

    @classmethod
    def create(cls, cond: Expr, msg: Optional[str] = None) -> AssertStmt:
        return cls(cond=cond, msg=msg)


@py_class("tilus.hidet.ir.AsmStmt", frozen=True, structural_eq="tree")
class AsmStmt(Stmt):
    template_string: str
    output_labels: Array[str]
    output_exprs: Array[Expr]
    input_labels: Array[str]
    input_exprs: Array[Expr]
    is_volatile: bool = False
    memory_fence: bool = False

    @classmethod
    def create(
        cls,
        template_string: str = "",
        outputs: Sequence[Tuple[str, Expr]] = (),
        inputs: Sequence[Tuple[str, Expr]] = (),
        is_volatile: bool = False,
        memory_fence: bool = False,
    ) -> AsmStmt:
        return cls(
            template_string=template_string,
            output_labels=tuple(pr[0] for pr in outputs),
            output_exprs=tuple(pr[1] for pr in outputs),
            input_labels=tuple(pr[0] for pr in inputs),
            input_exprs=tuple(pr[1] for pr in inputs),
            is_volatile=is_volatile,
            memory_fence=memory_fence,
        )


# Back-compat alias — some callers still import asm_stmt.
def asm_stmt(*args, **kwargs) -> AsmStmt:
    return AsmStmt.create(*args, **kwargs)


@py_class("tilus.hidet.ir.BlackBoxStmt", frozen=True, structural_eq="tree")
class BlackBoxStmt(Stmt):
    template_string: str
    exprs: Array[Expr]

    @classmethod
    def create(cls, template_string: str, *exprs: Expr) -> BlackBoxStmt:
        expected = template_string.count("{}")
        if expected != len(exprs):
            raise ValueError("Invalid template string: {} for {} args".format(template_string, len(exprs)))
        return cls(template_string=template_string, exprs=tuple(exprs))


@py_class("tilus.hidet.ir.SeqStmt", frozen=True, structural_eq="tree")
class SeqStmt(Stmt):
    seq: Array[Stmt]

    @classmethod
    def create(cls, seq: Sequence[Stmt]) -> SeqStmt:
        return cls(seq=tuple(seq))


@py_class("tilus.hidet.ir.LaunchKernelStmt", frozen=True, structural_eq="tree")
class LaunchKernelStmt(Stmt):
    _supported_targets: ClassVar[tuple[str, ...]] = ("cuda", "hip", "cpu")

    func_var: Var
    args: Array[Expr]
    grid_dim: Array[Expr]
    cluster_dim: Array[Expr]
    block_dim: Array[Expr]
    shared_mem_bytes: Expr
    target: str

    @classmethod
    def create(
        cls,
        func_var: Var,
        args: Sequence[Expr],
        grid_dim: Sequence[Expr],
        cluster_dim: Sequence[Expr],
        block_dim: Sequence[Expr],
        shared_mem_bytes: Expr,
        target: str,
    ) -> LaunchKernelStmt:
        assert func_var.name is not None
        if target not in cls._supported_targets:
            raise ValueError(f"Unsupported target: {target}")
        return cls(
            func_var=func_var,
            args=tuple(args),
            grid_dim=tuple(grid_dim),
            cluster_dim=tuple(cluster_dim),
            block_dim=tuple(block_dim),
            shared_mem_bytes=shared_mem_bytes,
            target=target,
        )


# ---------------------------------------------------------------------------
# ``asm`` — high-level inline-asm helper used by the primitive libraries.
# ---------------------------------------------------------------------------


def asm(
    template: str,
    *,
    outputs: Sequence[Any] = (),
    output_inputs: Sequence[Any] = (),
    inputs: Sequence[Any] = (),
    is_volatile: bool = False,
    memory_fence: bool = False,
) -> AsmStmt:
    from tilus.hidet.ir.tools import infer_type  # noqa: PLC0415

    if not isinstance(outputs, Sequence):
        raise TypeError("outputs must be a sequence")
    if not isinstance(output_inputs, Sequence):
        raise TypeError("output_inputs must be a sequence")
    if not isinstance(inputs, Sequence):
        raise TypeError("inputs must be a sequence")

    updated_outputs: List[Tuple[str, Expr]] = []
    updated_inputs: List[Tuple[str, Expr]] = []

    def get_register_type(expr: Expr) -> str:
        expr = convert(expr)
        expr_type = infer_type(expr)
        if isinstance(expr_type, DataType):
            if isinstance(expr, Constant):
                return "n"
            dtype2reg = {
                "float16": "h",
                "float32": "f",
                "bfloat16": "h",
                "float64": "d",
                "uint8": "h",
                "uint16": "h",
                "uint32": "r",
                "uint64": "l",
                "int8": "h",
                "int16": "h",
                "int32": "r",
                "int64": "l",
            }
            if expr_type.name not in dtype2reg:
                raise NotImplementedError("{}".format(expr_type))
            return dtype2reg[expr_type.name]
        if isinstance(expr_type, (PointerType, TensorPointerType)):
            return "l"
        raise ValueError("Can not deal with type {} in asm code.".format(expr_type))

    for output in outputs:
        constraint = "=" + get_register_type(output)
        updated_outputs.append((constraint, convert(output)))
    for output_input in output_inputs:
        constraint = "+" + get_register_type(output_input)
        updated_outputs.append((constraint, convert(output_input)))
    for x in inputs:
        constraint = get_register_type(x)
        updated_inputs.append((constraint, convert(x)))
    return AsmStmt.create(
        template_string=template,
        outputs=updated_outputs,
        inputs=updated_inputs,
        is_volatile=is_volatile,
        memory_fence=memory_fence,
    )


Int = Union[Expr, int]


def launch_kernel(
    func_var: Var,
    args: Sequence[Expr],
    grid_dim: Union[Sequence[Int], Int],
    block_dim: Union[Sequence[Int], Int],
    cluster_dim: Union[Sequence[Int], Int] = 1,
    shared_mem: Optional[Int] = 0,
    target: str = None,
) -> LaunchKernelStmt:
    launch_config: List[tuple] = []
    for dims in [grid_dim, cluster_dim, block_dim]:
        if not isinstance(dims, (list, tuple)):
            dims = [dims]
        dims = list(dims)
        if len(dims) > 3:
            raise ValueError("Grid/Cluster/Block dimension must be 3 or less.")
        while len(dims) < 3:
            dims.append(1)
        launch_config.append(tuple(convert(dims)))
    grid_dim_t, cluster_dim_t, block_dim_t = launch_config
    return LaunchKernelStmt.create(
        func_var=func_var,
        args=tuple(convert(a) for a in args),
        grid_dim=grid_dim_t,
        cluster_dim=cluster_dim_t,
        block_dim=block_dim_t,
        shared_mem_bytes=convert(shared_mem),
        target=target,
    )
