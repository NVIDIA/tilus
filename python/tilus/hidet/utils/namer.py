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
from collections import defaultdict
from typing import Iterable


class Namer:
    def __init__(self):
        self.name_id_clock = defaultdict(int)
        self.obj_name = {}
        #: Seeds survive :meth:`clear`. Populated by :meth:`seed` /
        #: :meth:`seed_global_symbols` for globally-unique symbols (CUDA
        #: builtins, function references) whose names must appear verbatim.
        self._seeds: dict = {}
        self.clear()

    def __call__(self, x):
        return self.get_name(x)

    def clear(self):
        self.name_id_clock.clear()
        self.obj_name.clear()
        # add keywords in target language
        keywords = ["const"]
        for kw in keywords:
            self.name_id_clock[kw] = 0
        # restore seeds
        for obj, name in self._seeds.items():
            self.obj_name[obj] = name
            if name not in self.name_id_clock:
                self.name_id_clock[name] = 0

    def get_name(self, e, hint=None):
        from tilus.hidet.ir.expr import Var

        if e in self.obj_name:
            return self.obj_name[e]
        if hint:
            orig_name = hint
        elif isinstance(e, Var) and e.name is not None:
            orig_name = e.name
        else:
            alias = {Var: "v"}
            orig_name = alias[type(e)] if type(e) in alias else type(e).__name__

        if orig_name in self.name_id_clock:
            name = orig_name
            while name in self.name_id_clock:
                self.name_id_clock[orig_name] += 1
                name = orig_name + "_" + str(self.name_id_clock[orig_name])
        else:
            self.name_id_clock[orig_name] = 0
            name = orig_name

        self.obj_name[e] = name
        return name

    def seed(self, obj, name: str) -> None:
        """Pre-register ``obj`` with a canonical ``name``.

        Future :meth:`get_name` calls on ``obj`` return ``name`` verbatim. The
        name is also reserved in ``name_id_clock`` so later colliding objects
        get suffixed (``name_1``, ``name_2``, ...). Seeds survive
        :meth:`clear`, so globally-unique symbols keep their canonical
        identifiers across function boundaries in codegen.
        """
        self._seeds[obj] = name
        self.obj_name[obj] = name
        if name not in self.name_id_clock:
            self.name_id_clock[name] = 0

    def seed_global_symbols(self, ir_module=None) -> None:
        """Pre-register all globally-unique symbols so codegen emits them verbatim.

        This covers:
          - registered CUDA / HIP primitive variables (threadIdx.x, blockIdx.y, ...)
          - primitive function references
          - optionally, the given IRModule's global vars and extern function refs

        Any later local Var whose name collides with one of these will be
        suffixed by :meth:`get_name` (``x``, ``x_1``, ...).
        """
        from tilus.hidet.ir.primitives.func import primitive_func_pool
        from tilus.hidet.ir.primitives.vars import registered_primitive_variables

        for name, var in registered_primitive_variables.items():
            self.seed(var, name)
        for name, entry in primitive_func_pool.name2func.items():
            self.seed(entry.var, name)
        if ir_module is not None:
            for name, var in ir_module.global_vars.items():
                self.seed(var, name)
            for name, var in ir_module.extern_functions.items():
                self.seed(var, name)

    @staticmethod
    def unique_name_among(name: str, existed_names: Iterable[str]) -> str:
        name_set = set(existed_names)
        if name not in name_set:
            return name
        else:
            i = 1
            while name + "_" + str(i) in name_set:
                i += 1
            return name + "_" + str(i)
