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
from typing import Dict, List

from tilus.hidet.ir.expr import Var
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.module import IRModule
from tilus.hidet.ir.tools.rewriter import rewrite
from tilus.hidet.ir.tools.util_functors import collect


def rename_funcs(ir_module: IRModule, rmap: Dict[str, str]) -> IRModule:
    """
    Rename functions in an IRModule.

    Parameters
    ----------
    ir_module: IRModule
        The IRModule.
    rmap: Dict[str, str]
        The renaming map.

    Returns
    -------
    ret: IRModule
        The renamed IRModule.
    """
    used_vars: List[Var] = collect(ir_module, node_types=Var)
    func_vars: List[Var] = [v for v in used_vars if v.type.is_func_type()]

    # rename the variables
    name2var: Dict[str, Var] = {}
    remap = {}
    for func_var in func_vars:
        if func_var.name in rmap:
            if func_var.name not in name2var:
                name2var[func_var.name] = Var(name=rmap[func_var.name], type=func_var.type)
            remap[func_var] = name2var[func_var.name]

    ir_module: IRModule = rewrite(ir_module, remap)

    # rename functions
    new_functions: Dict[str, Function] = {}
    for name, func in ir_module.functions.items():
        if name in rmap:
            new_functions[rmap[name]] = Function(
                name=rmap[name],
                params=func.params,
                body=func.body,
                ret_type=func.ret_type,
                kind=func.kind,
                attrs=func.attrs,
            )
        else:
            new_functions[name] = func

    # rename global vars
    global_vars: Dict[str, Var] = {}
    for name, var in ir_module.global_vars.items():
        if name in rmap:
            assert var.name == rmap[name]
            global_vars[rmap[name]] = var
        else:
            global_vars[name] = var

    ir_module.reset_funcs(functions=new_functions, global_vars=global_vars)
    return ir_module
