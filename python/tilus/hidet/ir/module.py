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
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import Dict, List, Sequence

from tilus.hidet.ir.expr import Var
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.node import Node
from tilus.hidet.ir.type import FuncType


class IRModule(Node):
    """
    The intermediate representation of tensor programs.

    An IRModule contains one or more functions. It is the basic compilation unit of hidet.
    """

    def __init__(
        self,
        functions=None,
        global_vars=None,
        namespace="",
        extern_functions: Dict[str, Var] = None,
        include_headers: List[str] = None,
        include_dirs: List[str] = None,
        linking_dirs: List[str] = None,
        linking_libs: List[str] = None,
        object_files: List[str] = None,
        task: "Task" = None,
    ):
        # the functions defined in this module
        self.functions: Dict[str, Function] = functions if functions else {}
        # the global variables defined in this module
        self.global_vars: Dict[str, Var] = global_vars if global_vars else {}
        # the namespace of the module, all the functions and the global variables will be defined in this namespace
        self.namespace: str = namespace
        # the external functions that are used in this module, the Var must have a function type
        self.extern_functions: Dict[str, Var] = {} if extern_functions is None else extern_functions
        # '#include ...' preprocessor directives
        self.include_headers: List[str] = include_headers if include_headers else []
        # flags that will be passed to the underlying compiler, can be used to add 3rd-party libraries
        self.include_dirs: List[str] = include_dirs if include_dirs else []  # -L flags
        self.linking_dirs: List[str] = linking_dirs if linking_dirs else []  # -I flags
        self.linking_libs: List[str] = linking_libs if linking_libs else []  # -l flags
        self.object_files: List[str] = object_files if object_files else []  # .o files
        self.task = task

        assert all(isinstance(func, Function) for func in self.functions.values()) and all(
            isinstance(var, Var) for var in self.global_vars.values()
        )

    def lookup_var(self, name):
        assert name in self.functions, (name, self.functions.keys())
        if name not in self.global_vars:
            func = self.functions[name]
            if isinstance(func, Function):
                self.global_vars[name] = Var(hint=None, type=FuncType.from_func(func), name=name)
            else:
                raise ValueError()

        return self.global_vars[name]

    def add_function(self, name, func: Function):
        if name in self.functions:
            raise ValueError("Function {} has already existed in module.".format(name))
        else:
            self.functions[name] = func

    def copy(self):
        return IRModule(
            functions=self.functions.copy(),
            global_vars=self.global_vars.copy(),
            namespace=self.namespace,
            extern_functions=self.extern_functions.copy(),
            include_headers=self.include_headers.copy(),
            include_dirs=self.include_dirs.copy(),
            linking_dirs=self.linking_dirs.copy(),
            linking_libs=self.linking_libs.copy(),
            object_files=self.object_files.copy(),
            task=self.task,
        )

    def build(self):
        """Build the IR module into a compiled module.

        Uses tilus's own lowering, codegen, and compilation pipeline.
        Returns a tvm_ffi module loaded from the compiled .so file.
        """
        import os
        import tempfile

        import tvm_ffi

        from tilus.drivers import build_ir_module

        output_dir = tempfile.mkdtemp(prefix="tilus_build_")
        build_ir_module(self, output_dir)
        return tvm_ffi.load_module(os.path.join(output_dir, "lib.so"))

    def reset_funcs(self, functions: Dict[str, Function] = None, global_vars: Dict[str, Var] = None):
        self.functions = functions if functions else {}
        self.global_vars = global_vars if global_vars else {}
        return self


def merge_ir_modules(modules: Sequence[IRModule]) -> IRModule:
    if len(modules) == 0:
        return IRModule()
    merged = modules[0].copy()
    for module in modules[1:]:
        if module.namespace != merged.namespace:
            raise ValueError("Cannot merge IRModules with different namespaces")
        # merge global vars
        for name, var in module.global_vars.items():
            if name in merged.global_vars:
                raise ValueError("Global variable {} has already existed in module.".format(name))
            merged.global_vars[name] = var
        # merge functions
        for name, func in module.functions.items():
            if name in merged.functions:
                raise ValueError("Function {} has already existed in module.".format(name))
            merged.functions[name] = func
        # merge extern functions
        for name, var in module.extern_functions.items():
            if name in merged.extern_functions:
                continue
            merged.extern_functions[name] = var

        # merge include headers, include_dirs, linking_dirs, linking_libs, object_files
        merged.include_headers.extend(
            [header for header in module.include_headers if header not in merged.include_dirs]
        )
        merged.include_dirs.extend([dir for dir in module.include_dirs if dir not in merged.include_dirs])
        merged.linking_dirs.extend([dir for dir in module.linking_dirs if dir not in merged.linking_dirs])
        merged.linking_libs.extend([lib for lib in module.linking_libs if lib not in merged.linking_libs])
        merged.object_files.extend([file for file in module.object_files if file not in merged.object_files])

    return merged
