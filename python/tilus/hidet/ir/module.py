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

from tilus.hidet.ir.expr import Var
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.node import Node
from tilus.hidet.ir.type import FuncType


class IRModule(Node):
    """The intermediate representation of tensor programs.

    Immutable by convention: all update operations return a new ``IRModule``. Consumers
    must never mutate ``functions`` / ``global_vars`` / ``extern_functions`` / the list
    fields in-place — use :meth:`with_functions`, :meth:`with_removed_functions`,
    :meth:`merge`, or construct a new ``IRModule``.
    """

    def __init__(
        self,
        functions: Optional[Dict[str, Function]] = None,
        global_vars: Optional[Dict[str, Var]] = None,
        namespace: str = "",
        extern_functions: Optional[Dict[str, Var]] = None,
        include_headers: Optional[List[str]] = None,
        include_dirs: Optional[List[str]] = None,
        linking_dirs: Optional[List[str]] = None,
        linking_libs: Optional[List[str]] = None,
        object_files: Optional[List[str]] = None,
    ):
        self.functions: Dict[str, Function] = dict(functions) if functions else {}
        self.global_vars: Dict[str, Var] = dict(global_vars) if global_vars else {}
        # Ensure every defined function has a corresponding global_var entry.
        for name, func in self.functions.items():
            if name not in self.global_vars:
                self.global_vars[name] = Var(name=name, type=FuncType.from_func(func))
        self.namespace: str = namespace
        self.extern_functions: Dict[str, Var] = dict(extern_functions) if extern_functions else {}
        self.include_headers: List[str] = list(include_headers) if include_headers else []
        self.include_dirs: List[str] = list(include_dirs) if include_dirs else []
        self.linking_dirs: List[str] = list(linking_dirs) if linking_dirs else []
        self.linking_libs: List[str] = list(linking_libs) if linking_libs else []
        self.object_files: List[str] = list(object_files) if object_files else []

        assert all(isinstance(func, Function) for func in self.functions.values()) and all(
            isinstance(var, Var) for var in self.global_vars.values()
        )

    def lookup_var(self, name: str) -> Var:
        assert name in self.global_vars, (name, list(self.global_vars.keys()))
        return self.global_vars[name]

    def copy(self) -> IRModule:
        return IRModule(
            functions=self.functions,
            global_vars=self.global_vars,
            namespace=self.namespace,
            extern_functions=self.extern_functions,
            include_headers=self.include_headers,
            include_dirs=self.include_dirs,
            linking_dirs=self.linking_dirs,
            linking_libs=self.linking_libs,
            object_files=self.object_files,
        )

    def build(self):
        """Build the IR module into a compiled module.

        Uses tilus's own lowering, codegen, and compilation pipeline.
        Returns a callable CompiledModule backed by the compiled .so file.
        """
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
    ) -> IRModule:
        """Return a new IRModule with replaced ``functions`` and (optionally) ``global_vars``.

        When ``global_vars`` is omitted, only the entries matching retained function names
        are kept and fresh entries are synthesized for newly added functions.
        """
        new_functions = dict(functions) if functions is not None else dict(self.functions)
        if global_vars is not None:
            new_global_vars = dict(global_vars)
        else:
            new_global_vars = {name: var for name, var in self.global_vars.items() if name in new_functions}
        return IRModule(
            functions=new_functions,
            global_vars=new_global_vars,
            namespace=self.namespace,
            extern_functions=self.extern_functions,
            include_headers=self.include_headers,
            include_dirs=self.include_dirs,
            linking_dirs=self.linking_dirs,
            linking_libs=self.linking_libs,
            object_files=self.object_files,
        )

    def with_added_functions(self, functions: Dict[str, Function]) -> IRModule:
        """Return a new IRModule with the given functions added (duplicates raise)."""
        new_functions = dict(self.functions)
        for name, func in functions.items():
            if name in new_functions:
                raise ValueError(f"Function {name} has already existed in module.")
            new_functions[name] = func
        return self.with_functions(functions=new_functions, global_vars=None)

    def with_removed_functions(self, names: Iterable[str]) -> IRModule:
        """Return a new IRModule with the given function names (and their global_vars) removed."""
        remove = set(names)
        new_functions = {name: func for name, func in self.functions.items() if name not in remove}
        new_global_vars = {name: var for name, var in self.global_vars.items() if name not in remove}
        return self.with_functions(functions=new_functions, global_vars=new_global_vars)


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
