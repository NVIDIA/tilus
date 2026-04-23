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
from typing import Dict, Sequence, Tuple, Union

from tilus.hidet.ir.builders import FunctionBuilder
from tilus.hidet.ir.dtypes import int32
from tilus.hidet.ir.expr import Expr, Var
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.module import IRModule
from tilus.hidet.ir.stmt import LaunchKernelStmt
from tilus.hidet.ir.tools import rewrite, simplify
from tilus.hidet.transforms.base import Pass


def _normalize_dim3(dim3: Union[int, Expr, Sequence[Union[int, Expr]]]) -> Tuple[Expr, Expr, Expr]:
    if isinstance(dim3, int):
        return int32(dim3), int32(1), int32(1)
    elif isinstance(dim3, Expr):
        return simplify(dim3), int32(1), int32(1)
    elif isinstance(dim3, Sequence):
        dim3 = [simplify(int32(v)) for v in dim3]
        assert len(dim3) <= 3
        while len(dim3) < 3:
            dim3.append(int32(1))
        return dim3[0], dim3[1], dim3[2]
    else:
        raise TypeError("Unsupported type: {}".format(type(dim3)))


def _rewrite_dim3(dim3: Tuple[Expr, Expr, Expr], param2arg: Dict[Expr, Expr]) -> Tuple[Expr, Expr, Expr]:
    return rewrite(dim3[0], param2arg), rewrite(dim3[1], param2arg), rewrite(dim3[2], param2arg)


def add_launch_func(ir_module: IRModule, kernel_func: Function) -> IRModule:
    with FunctionBuilder(name="launch", kind="public") as fb:
        params = [Var(param.name, param.type) for param in kernel_func.params]
        param_remap = {a: b for a, b in zip(kernel_func.params, params)}
        fb.extend_params(params)

        func_var = ir_module.lookup_var(kernel_func.name)

        if kernel_func.kind == "cuda_kernel":
            from tilus.hidet.lang.cuda import set_kernel_max_dynamic_smem_bytes

            dsb = kernel_func.attrs.dynamic_smem_bytes
            shared_memory_bytes: Expr = rewrite(simplify(dsb if dsb is not None else int32(0)), param_remap)
            with fb.if_then(shared_memory_bytes > 48 * 1024):
                fb += set_kernel_max_dynamic_smem_bytes(func_var, shared_memory_bytes)
            cluster_dim = kernel_func.attrs.cluster_dim if kernel_func.attrs.cluster_dim is not None else 1
            fb += LaunchKernelStmt.create(
                func_var,
                params,
                grid_dim=rewrite(_normalize_dim3(kernel_func.attrs.grid_dim), param_remap),
                cluster_dim=rewrite(_normalize_dim3(cluster_dim), param_remap),
                block_dim=rewrite(_normalize_dim3(kernel_func.attrs.block_dim), param_remap),
                shared_mem=shared_memory_bytes,
                target="cuda",
            )
        elif kernel_func.kind == "hip_kernel":
            dsb = kernel_func.attrs.dynamic_smem_bytes
            shared_memory_bytes: Expr = rewrite(simplify(dsb if dsb is not None else int32(0)), param_remap)

            fb += LaunchKernelStmt.create(
                func_var,
                params,
                grid_dim=rewrite(_normalize_dim3(kernel_func.attrs.grid_dim), param_remap),
                cluster_dim=(int32.one, int32.one, int32.one),
                block_dim=rewrite(_normalize_dim3(kernel_func.attrs.block_dim), param_remap),
                shared_mem=shared_memory_bytes,
                target="hip",
            )
        elif kernel_func.kind == "cpu_kernel":
            fb += func_var(*params)
        else:
            raise NotImplementedError("Unsupported function kind: {}".format(kernel_func.kind))

    launch: Function = fb.func
    return ir_module.with_added_functions({launch.name: launch})


class GenerateLaunchFuncPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        if any(func.name.startswith("launch") for func in ir_module.functions.values() if func.kind == "public"):
            return ir_module

        # Find existing codegen host functions (public, not "launch")
        host_funcs = [
            (name, func)
            for name, func in ir_module.functions.items()
            if func.kind == "public" and not name.startswith("launch")
        ]

        if len(host_funcs) == 1:
            # Single host function: rename it to "launch" so the runtime can find it.
            # This avoids creating a duplicate function.
            old_name, host_func = host_funcs[0]
            renamed = Function(
                name="launch",
                params=host_func.params,
                body=host_func.body,
                ret_type=host_func.ret_type,
                kind=host_func.kind,
                attrs=host_func.attrs,
            )
            return ir_module.with_removed_functions([old_name]).with_added_functions({"launch": renamed})

        # Multiple or no host functions: generate a launch function from the kernel
        kernel_functions: Dict[str, Function] = {
            name: func
            for name, func in ir_module.functions.items()
            if func.kind in ["cuda_kernel", "hip_kernel", "cpu_kernel"]
        }
        if len(kernel_functions) == 0:
            return ir_module
        if len(kernel_functions) > 1:
            raise NotImplementedError("Can only handle one kernel function in a module")
        kernel_func = next(iter(kernel_functions.values()))
        return add_launch_func(ir_module, kernel_func)


def generate_launch_func(ir_module: IRModule) -> IRModule:
    return GenerateLaunchFuncPass().process_module(ir_module)


def generate_launch_func_pass() -> Pass:
    return GenerateLaunchFuncPass()
