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

from typing import Dict, Iterable, List, Optional, Sequence

from tvm_ffi.dataclasses import field, py_class

from tilus.hidet.ir.expr import Var
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.node import Node
from tilus.hidet.ir.type import FuncType


@py_class("tilus.hidet.ir.IRModule", frozen=True)
class IRModule(Node):
    """The intermediate representation of tensor programs.

    Frozen py_class: all update operations construct a new ``IRModule``.
    Uses ``ffi.Map`` (immutable) for the keyed collections and ``ffi.List``
    for the string lists.
    """

    functions: dict[str, Function] = field(default_factory=dict)
    global_vars: dict[str, Var] = field(default_factory=dict)
    namespace: str = ""
    extern_functions: dict[str, Var] = field(default_factory=dict)
    include_headers: list[str] = field(default_factory=list)
    include_dirs: list[str] = field(default_factory=list)
    linking_dirs: list[str] = field(default_factory=list)
    linking_libs: list[str] = field(default_factory=list)
    object_files: list[str] = field(default_factory=list)

    def lookup_var(self, name: str) -> Var:
        assert name in self.global_vars, (name, list(self.global_vars.keys()))
        return self.global_vars[name]

    def copy(self) -> "IRModule":
        return IRModule(
            functions=dict(self.functions),
            global_vars=dict(self.global_vars),
            namespace=self.namespace,
            extern_functions=dict(self.extern_functions),
            include_headers=list(self.include_headers),
            include_dirs=list(self.include_dirs),
            linking_dirs=list(self.linking_dirs),
            linking_libs=list(self.linking_libs),
            object_files=list(self.object_files),
        )

    def build(self):
        """Build the IR module into a compiled module via tilus's pipeline."""
        import os
        import tempfile

        from tilus.drivers import build_ir_module
        from tilus.runtime.compiled_program import CompiledModule

        output_dir = tempfile.mkdtemp(prefix="tilus_build_")
        build_ir_module(self, output_dir)
        return CompiledModule(os.path.join(output_dir, "lib.so"))

    # ---- functional updates ----

    def with_functions(
        self,
        functions: Optional[Dict[str, Function]] = None,
        global_vars: Optional[Dict[str, Var]] = None,
    ) -> "IRModule":
        """Return a new IRModule with replaced ``functions`` and (optionally) ``global_vars``.

        When ``global_vars`` is omitted, entries matching retained function names
        are kept and fresh entries are synthesized for newly added functions.
        """
        new_functions = dict(functions) if functions is not None else dict(self.functions)
        if global_vars is not None:
            new_global_vars = dict(global_vars)
        else:
            new_global_vars = {name: var for name, var in self.global_vars.items() if name in new_functions}
        # Synthesize missing global_vars for newly added functions.
        for name, func in new_functions.items():
            if name not in new_global_vars:
                new_global_vars[name] = Var(name=name, type=FuncType.from_func(func))
        return IRModule(
            functions=new_functions,
            global_vars=new_global_vars,
            namespace=self.namespace,
            extern_functions=dict(self.extern_functions),
            include_headers=list(self.include_headers),
            include_dirs=list(self.include_dirs),
            linking_dirs=list(self.linking_dirs),
            linking_libs=list(self.linking_libs),
            object_files=list(self.object_files),
        )

    def with_added_functions(self, functions: Dict[str, Function]) -> "IRModule":
        new_functions = dict(self.functions)
        for name, func in functions.items():
            if name in new_functions:
                raise ValueError(f"Function {name} has already existed in module.")
            new_functions[name] = func
        return self.with_functions(functions=new_functions, global_vars=None)

    def with_removed_functions(self, names: Iterable[str]) -> "IRModule":
        remove = set(names)
        new_functions = {name: func for name, func in self.functions.items() if name not in remove}
        new_global_vars = {name: var for name, var in self.global_vars.items() if name not in remove}
        return self.with_functions(functions=new_functions, global_vars=new_global_vars)


def ir_module(
    functions: Optional[Dict[str, Function]] = None,
    global_vars: Optional[Dict[str, Var]] = None,
    namespace: str = "",
    extern_functions: Optional[Dict[str, Var]] = None,
    include_headers: Optional[List[str]] = None,
    include_dirs: Optional[List[str]] = None,
    linking_dirs: Optional[List[str]] = None,
    linking_libs: Optional[List[str]] = None,
    object_files: Optional[List[str]] = None,
) -> IRModule:
    """Factory that pre-normalizes defaults and auto-synthesizes missing
    ``global_vars`` entries for defined functions."""
    functions = dict(functions) if functions else {}
    global_vars = dict(global_vars) if global_vars else {}
    for name, func in functions.items():
        if name not in global_vars:
            global_vars[name] = Var(name=name, type=FuncType.from_func(func))
    return IRModule(
        functions=functions,
        global_vars=global_vars,
        namespace=namespace,
        extern_functions=dict(extern_functions) if extern_functions else {},
        include_headers=list(include_headers) if include_headers else [],
        include_dirs=list(include_dirs) if include_dirs else [],
        linking_dirs=list(linking_dirs) if linking_dirs else [],
        linking_libs=list(linking_libs) if linking_libs else [],
        object_files=list(object_files) if object_files else [],
    )


def merge_ir_modules(modules: Sequence[IRModule]) -> IRModule:
    if len(modules) == 0:
        return IRModule()
    base = modules[0]
    namespace = base.namespace
    functions: Dict[str, Function] = dict(base.functions)
    global_vars: Dict[str, Var] = dict(base.global_vars)
    extern_functions: Dict[str, Var] = dict(base.extern_functions)
    include_headers: List[str] = list(base.include_headers)
    include_dirs: List[str] = list(base.include_dirs)
    linking_dirs: List[str] = list(base.linking_dirs)
    linking_libs: List[str] = list(base.linking_libs)
    object_files: List[str] = list(base.object_files)

    for module in modules[1:]:
        if module.namespace != namespace:
            raise ValueError("Cannot merge IRModules with different namespaces")
        for name, var in module.global_vars.items():
            if name in global_vars:
                raise ValueError(f"Global variable {name} has already existed in module.")
            global_vars[name] = var
        for name, func in module.functions.items():
            if name in functions:
                raise ValueError(f"Function {name} has already existed in module.")
            functions[name] = func
        for name, var in module.extern_functions.items():
            if name not in extern_functions:
                extern_functions[name] = var
        include_headers.extend(h for h in module.include_headers if h not in include_headers)
        include_dirs.extend(d for d in module.include_dirs if d not in include_dirs)
        linking_dirs.extend(d for d in module.linking_dirs if d not in linking_dirs)
        linking_libs.extend(lib for lib in module.linking_libs if lib not in linking_libs)
        object_files.extend(f for f in module.object_files if f not in object_files)

    return IRModule(
        functions=functions,
        global_vars=global_vars,
        namespace=namespace,
        extern_functions=extern_functions,
        include_headers=include_headers,
        include_dirs=include_dirs,
        linking_dirs=linking_dirs,
        linking_libs=linking_libs,
        object_files=object_files,
    )
