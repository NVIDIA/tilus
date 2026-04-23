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
from dataclasses import dataclass
from typing import Generic, List, Optional, Sequence, Tuple, TypeVar, Union, cast

from tilus.hidet.ir.expr import Expr, Var, convert, var
from tilus.hidet.ir.stmt import (
    AssertStmt,
    AssignStmt,
    BreakStmt,
    BufferStoreStmt,
    DeclareScope,
    DeclareStmt,
    EvaluateStmt,
    ForStmt,
    ForStmtAttr,
    IfStmt,
    LetStmt,
    ReturnStmt,
    SeqStmt,
    Stmt,
    WhileStmt,
)
from tilus.hidet.ir.type import BaseType


@dataclass
class _Frame:
    """An open scope in the builder. Inherit and implement :meth:`build`.

    Subclasses either produce a final Stmt from the accumulated body, or participate
    in an in-flight if/elif/else ladder handled directly by the StmtBuilder.
    """

    def build(self, body: Stmt) -> Stmt:
        raise NotImplementedError


@dataclass
class _ForFrame(_Frame):
    loop_var: Var
    extent: Expr
    attr: ForStmtAttr

    def build(self, body: Stmt) -> ForStmt:
        return ForStmt.create(self.loop_var, self.extent, body, attr=self.attr)


@dataclass
class _LetFrame(_Frame):
    bind_vars: Tuple[Var, ...]
    bind_values: Tuple[Expr, ...]

    def build(self, body: Stmt) -> LetStmt:
        return LetStmt.create(self.bind_vars, self.bind_values, body)


@dataclass
class _WhileFrame(_Frame):
    cond: Expr

    def build(self, body: Stmt) -> WhileStmt:
        return WhileStmt.create(self.cond, body)


@dataclass
class _IfThenFrame(_Frame):
    cond: Expr


@dataclass
class _ElseIfFrame(_Frame):
    cond: Expr


@dataclass
class _OtherwiseFrame(_Frame):
    pass


class StmtScope:
    """Context manager that opens one or more scopes on a :class:`StmtBuilder`."""

    def __init__(self, sb: "StmtBuilder", frames: Sequence[_Frame], ret=None):
        self.sb: StmtBuilder = sb
        self.frames: List[_Frame] = list(frames)
        self.ret = ret

    def __enter__(self):
        for frame in self.frames:
            self.sb._enter_scope(frame)
        return self.ret

    def __exit__(self, exc_type, exc_val, exc_tb):
        for _ in self.frames:
            self.sb._exit_scope()


class StmtBuilder:
    """Build statements imperatively, but construct every IR node immutably bottom-up.

    The builder maintains three parallel stacks, one entry per open scope:

    - ``scope_stack[i]`` — completed sibling statements accumulated in scope ``i``.
    - ``frame_stack[i-1]`` — description of the scope-opening statement (``_Frame``).
      Has one fewer entry than ``scope_stack`` because the outermost scope has no frame.
    - ``pending_if_stack[i]`` — in-flight if/elif ladder in scope ``i``. Held as an
      ``IfStmt`` whose deepest ``else_body`` may still be ``None``; flushed when
      any non-``else_if``/``otherwise`` stmt arrives or when the enclosing scope ends.
    """

    def __init__(self):
        self.scope_stack: List[List[Stmt]] = [[]]
        self.frame_stack: List[_Frame] = []
        self.pending_if_stack: List[Optional[IfStmt]] = [None]

    def __iadd__(self, other: Union[Stmt, Expr, Sequence[Stmt]]):
        assert isinstance(other, (Stmt, Expr, list, tuple))
        self.append(other)
        return self

    @staticmethod
    def _name_index_vars(num_vars: int) -> List[str]:
        predefined = ["i", "j", "k", "p", "q", "r", "s", "u", "v"]
        if num_vars <= len(predefined):
            return predefined[:num_vars]
        return [f"i{idx}" for idx in range(num_vars)]

    # ---- singleton statements ----

    def declare(self, v: Var, init: Optional[Expr] = None, scope=None):
        self.append(DeclareStmt.create(v, convert(init), scope=scope))
        return v

    def declare_var(
        self, name: str, tp: BaseType, init: Optional[Expr] = None, scope: Optional[DeclareScope] = None
    ) -> Var:
        v = var(name, tp)
        self.append(DeclareStmt.create(v, init=convert(init), scope=scope))
        return v

    def buffer_store(self, buf: Expr, indices: Sequence[Union[Expr, int]], value: Expr):
        self.append(BufferStoreStmt.create(buf, [convert(i) for i in indices], convert(value)))

    def assign(self, dst: Var, value: Expr):
        self.append(AssignStmt.create(dst, convert(value)))

    def assertion(self, cond: Union[Expr, bool], msg: str) -> None:
        self.append(AssertStmt.create(convert(cond), msg))

    def comment(self, comment_string: str, style: str = "//") -> None:
        from tilus.hidet.ir.primitives.debug import comment

        self.append(comment(comment_string, style=style))

    def brk(self):
        self.append(BreakStmt())

    def ret(self, value: Optional[Expr] = None):
        self.append(ReturnStmt(value))

    # ---- scope-opening statements ----

    def let(self, v: Union[str, Var], value: Union[int, Expr]) -> StmtScope:
        if isinstance(v, str):
            v = var(v)
        return StmtScope(self, frames=[_LetFrame((v,), (convert(value),))], ret=v)

    def lets(self, bind_vars: Sequence[Union[str, Var]], values: Sequence[Union[int, Expr]]) -> StmtScope:
        assert len(bind_vars) == len(values)
        resolved_vars = tuple(var(v) if isinstance(v, str) else v for v in bind_vars)
        resolved_values = tuple(convert(value) for value in values)
        return StmtScope(self, frames=[_LetFrame(resolved_vars, resolved_values)], ret=list(resolved_vars))

    def for_loop(self, v: Union[str, Var], extent: Union[int, Expr], attr: str = ".") -> StmtScope:
        if isinstance(v, str):
            v = var(v)
        parsed_attr = ForStmtAttr.parse(attr, num_loops=1)[0]
        return StmtScope(self, frames=[_ForFrame(v, convert(extent), parsed_attr)], ret=v)

    def if_then(self, cond: Union[bool, Expr]) -> StmtScope:
        return StmtScope(self, frames=[_IfThenFrame(convert(cond))], ret=None)

    def else_if(self, cond: Union[bool, Expr]) -> StmtScope:
        return StmtScope(self, frames=[_ElseIfFrame(convert(cond))], ret=None)

    def otherwise(self) -> StmtScope:
        return StmtScope(self, frames=[_OtherwiseFrame()], ret=None)

    def for_grid(self, shape: Sequence[Union[Expr, int]]) -> StmtScope:
        iter_names = self._name_index_vars(len(shape))
        iter_vars = [var(name) for name in iter_names]
        frames = [_ForFrame(iv, convert(extent), ForStmtAttr()) for iv, extent in zip(iter_vars, shape)]
        return StmtScope(self, frames=frames, ret=iter_vars)

    def for_range(self, extent: Union[Expr, int], *, attr: Optional[Union[str, ForStmtAttr]] = None) -> StmtScope:
        iter_var = var("i")
        if isinstance(attr, str):
            attr_obj = ForStmtAttr.parse(attr, num_loops=1)[0]
        elif isinstance(attr, ForStmtAttr):
            attr_obj = attr
        else:
            attr_obj = ForStmtAttr()
        return StmtScope(self, frames=[_ForFrame(iter_var, convert(extent), attr_obj)], ret=iter_var)

    def while_loop(self, cond: Expr) -> StmtScope:
        return StmtScope(self, frames=[_WhileFrame(convert(cond))], ret=None)

    # ---- core ----

    def append(self, stmt: Union[Stmt, Expr, Sequence[Stmt], None]) -> None:
        if stmt is None:
            return
        if isinstance(stmt, (Stmt, Expr)):
            self._flush_pending_if()
            if isinstance(stmt, Expr):
                stmt = EvaluateStmt.create(stmt)
            self.scope_stack[-1].append(stmt)
        else:
            assert isinstance(stmt, (tuple, list))
            for s in stmt:
                self.append(s)

    def _flush_pending_if(self) -> None:
        pending = self.pending_if_stack[-1]
        if pending is not None:
            self.scope_stack[-1].append(pending)
            self.pending_if_stack[-1] = None

    def _enter_scope(self, frame: _Frame) -> None:
        if isinstance(frame, (_ElseIfFrame, _OtherwiseFrame)):
            if self.pending_if_stack[-1] is None:
                raise RuntimeError(f"{type(frame).__name__[1:-5]}() must follow if_then() or else_if()")
        else:
            self._flush_pending_if()
        self.frame_stack.append(frame)
        self.scope_stack.append([])
        self.pending_if_stack.append(None)

    def _exit_scope(self) -> None:
        self._flush_pending_if()
        body = SeqStmt.create(self.scope_stack.pop())
        self.pending_if_stack.pop()
        frame = self.frame_stack.pop()
        if isinstance(frame, _IfThenFrame):
            self.pending_if_stack[-1] = IfStmt.create(frame.cond, body, None)
        elif isinstance(frame, _ElseIfFrame):
            prev = self.pending_if_stack[-1]
            assert prev is not None  # enforced at _enter_scope
            self.pending_if_stack[-1] = _attach_else(prev, IfStmt.create(frame.cond, body, None))
        elif isinstance(frame, _OtherwiseFrame):
            prev = self.pending_if_stack[-1]
            assert prev is not None  # enforced at _enter_scope
            self.pending_if_stack[-1] = _attach_else(prev, body)
        else:
            self.scope_stack[-1].append(frame.build(body))

    def finish(self) -> SeqStmt:
        self._flush_pending_if()
        assert len(self.scope_stack) == 1 and not self.frame_stack, "finish() called with open scopes"
        return SeqStmt.create(self.scope_stack.pop())


def _attach_else(prev_if: IfStmt, new_else: Stmt) -> IfStmt:
    """Return a new IfStmt tree with ``new_else`` attached to the innermost ``None`` else slot."""
    if prev_if.else_body is None:
        return IfStmt.create(prev_if.cond, prev_if.then_body, new_else)
    assert isinstance(prev_if.else_body, IfStmt), "otherwise() must be the last entry in an if-chain"
    return IfStmt.create(prev_if.cond, prev_if.then_body, _attach_else(prev_if.else_body, new_else))


T = TypeVar("T")


class TypedStmtScope(StmtScope, Generic[T]):
    def __enter__(self) -> T:
        return cast(T, super().__enter__())

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)


class TypedStmtBuilder(StmtBuilder):
    def for_range(
        self, extent: Union[Expr, int], *, attr: Optional[Union[str, ForStmtAttr]] = None
    ) -> TypedStmtScope[Var]:
        return cast(TypedStmtScope[Var], super().for_range(extent, attr=attr))

    def for_grid(self, shape: Sequence[Union[Expr, int]]) -> TypedStmtScope[list[Var]]:
        return cast(TypedStmtScope[list[Var]], super().for_grid(shape))
