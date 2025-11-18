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

from typing import Any, Optional, Union

from hidet.ir.expr import Var

import tilus.lang.constructs.loops
from tilus.ir.builders import StmtBuilder
from tilus.ir.tensor import Tensor

_current_scope: Optional[Scope] = None
_scope_stack: list[Scope] = []


class Scope:
    def __init__(self) -> None:
        self.parent: Optional[Scope] = None
        self.name2var: dict[str, Var] = {}
        self.name2value: dict[str, Tensor] = {}
        self.name2host_var: dict[str, Any] = {}

    def __enter__(self) -> Scope:
        global _current_scope
        self.parent = _current_scope
        _current_scope = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return
        global _current_scope
        assert _current_scope is self
        _current_scope = self.parent

    def bind(self, name: str, var_or_value: Var | Tensor | Any) -> None:
        if name in self.name2var or name in self.name2value or name in self.name2host_var:
            if name == "_":
                # we allow re-binding to '_' variable to facilitate discarding assignments
                return
            raise RuntimeError(f'Variable "{name}" has already been defined in the current scope.')
        if isinstance(var_or_value, Var):
            self.name2var[name] = var_or_value
        elif isinstance(var_or_value, Tensor):
            self.name2value[name] = var_or_value
        else:
            self.name2host_var[name] = var_or_value

    def lookup(self, name: str) -> Var | Tensor | Any | None:
        if name in self.name2var:
            return self.name2var[name]
        if name in self.name2value:
            return self.name2value[name]
        if name in self.name2host_var:
            return self.name2host_var[name]
        if self.parent:
            return self.parent.lookup(name)
        return None


class ScopedProgramBuilder(StmtBuilder):
    def __init__(self):
        super().__init__()
        self.builtin_scope: Scope = Scope()

        # initialize the builtin scope
        self.builtin_scope.bind("range", tilus.lang.constructs.loops.range)

        # set the builtin scope as the top-level scope
        global _current_scope
        _current_scope = self.builtin_scope

    @property
    def current_scope(self) -> Scope:
        """Get the current scope."""
        global _current_scope
        assert _current_scope is not None, "No current scope."
        return _current_scope

    def scope(self) -> Scope:
        """Create a new scope."""
        return Scope()

    def bind(self, name: str, var_or_value: Var | Tensor | Any) -> None:
        """Bind a name to a variable or value in the current scope."""
        self.current_scope.bind(name, var_or_value)

    def lookup(self, name: str) -> Union[Var, Tensor, Any]:
        """Lookup a name in the current scope chain."""
        return self.current_scope.lookup(name)

    def dump_and_push_scopes(self) -> None:
        """Dump the current scope to the stack and push the builtin scope as the current scope."""
        global _scope_stack, _current_scope
        assert _current_scope is not None
        _scope_stack.append(_current_scope)
        _current_scope = self.builtin_scope

    def pop_and_restore_scopes(self) -> None:
        """Pop the last scope from the stack and restore it as the current scope."""
        global _scope_stack, _current_scope
        assert _current_scope is self.builtin_scope, "Scope mismatch when popping and restoring scopes."
        assert len(_scope_stack) > 0
        _current_scope = _scope_stack.pop()
