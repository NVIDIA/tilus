# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Lower fastdiv(a, b) calls to use precomputed magic multiplier and shift.

This pass:
1. Scans the kernel function for fastdiv(a, b) calls
2. Checks that b is a grid-constant expression (composed only of kernel parameters and grid dimensions)
3. For each unique grid-constant b, adds two new kernel parameters (multiplier, shift)
4. Replaces fastdiv(a, b) with fastdiv_runtime(a, multiplier, shift) in the kernel
5. Adds host-side precomputation of (multiplier, shift) from b in the launch function
"""

from typing import Dict, List, Optional, Set, Tuple

from tilus.hidet.ir.dtypes import int32, uint32
from tilus.hidet.ir.expr import Call, Cast, Expr, Var
from tilus.hidet.ir.func import Function
from tilus.hidet.ir.functors import IRRewriter
from tilus.hidet.ir.module import IRModule
from tilus.hidet.ir.primitives import is_primitive_function
from tilus.hidet.ir.primitives.cuda.fast_divmod import fastdiv_precompute_m, fastdiv_precompute_s, fastdiv_runtime
from tilus.hidet.ir.stmt import DeclareStmt, LaunchKernelStmt, LetStmt, SeqStmt
from tilus.hidet.ir.tools import collect, rewrite
from tilus.hidet.ir.tools.printer import IRPrinter
from tilus.hidet.transforms.base import Pass


class GridConstantTracker:
    """Tracks which variables are grid-constant (known at kernel launch time).

    A variable is grid-constant if it is a kernel parameter or is defined as an expression
    composed entirely of other grid-constant variables.
    """

    def __init__(self, kernel_params: List[Var]):
        self.grid_constants: Set[Var] = set(kernel_params)
        self.var2expr: Dict[Var, Expr] = {p: p for p in kernel_params}

    def bind(self, var: Var, value: Expr) -> None:
        """Record a variable binding. If the value is grid-constant, mark the variable as grid-constant."""
        used_vars = collect(value, Var, stop_when_found=True)
        if all(v in self.grid_constants for v in used_vars):
            self.grid_constants.add(var)
            self.var2expr[var] = rewrite(value, self.var2expr)

    def is_grid_constant(self, expr: Expr) -> bool:
        """Check if an expression is composed entirely of grid-constant variables."""
        used_vars = collect(expr, Var, stop_when_found=True)
        return all(v in self.grid_constants for v in used_vars)

    def expand(self, expr: Expr) -> Expr:
        """Expand an expression by replacing all variables with their grid-constant definitions."""
        return rewrite(expr, self.var2expr)


class FastDivRewriter(IRRewriter):
    """First pass: collect all fastdiv calls and track grid-constant expressions."""

    def __init__(self, tracker: GridConstantTracker):
        super().__init__()
        self.tracker = tracker
        self.printer = IRPrinter()
        # Map from expanded grid-constant divisor expression (as string) to (divisor_expr, m_var, s_var)
        self.divisor_map: Dict[str, Tuple[Expr, Var, Var]] = {}
        self.found_fastdiv = False

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        if stmt.init is not None:
            self.tracker.bind(stmt.var, stmt.init)
        return IRRewriter.visit_DeclareStmt(self, stmt)

    def visit_LetStmt(self, stmt: LetStmt):
        for var, value in zip(stmt.bind_vars, stmt.bind_values):
            self.tracker.bind(var, value)
        return IRRewriter.visit_LetStmt(self, stmt)

    def visit_Call(self, e: Call):
        if e.func_var.name == "cuda_fastdiv":
            self.found_fastdiv = True
            a, b = e.args
            # Check b is grid-constant
            if not self.tracker.is_grid_constant(b):
                raise ValueError(
                    "fastdiv divisor must be a grid-constant expression (composed only of kernel "
                    "parameters and grid dimensions), but got: {}".format(b)
                )
            # Expand b to canonical form for dedup
            expanded_b = self.tracker.expand(b)
            key = str(self.printer(expanded_b))
            if key not in self.divisor_map:
                m_var = Var("fast_div_m", type=int32)
                s_var = Var("fast_div_s", type=int32)
                self.divisor_map[key] = (expanded_b, m_var, s_var)
            _, m_var, s_var = self.divisor_map[key]
            # Replace fastdiv(a, b) with fastdiv_runtime(a, m, s)
            new_a = self.visit(a)
            return fastdiv_runtime(new_a, m_var, s_var)
        return IRRewriter.visit_Call(self, e)


class LaunchStmtRewriter(IRRewriter):
    """Rewrites LaunchKernelStmt to insert precompute stmts and extra args.

    Only rewrites launch statements that target a kernel in the provided mapping.
    """

    def __init__(self, kernel_launch_info: Dict[str, Tuple[Function, List, List[Var]]]):
        """
        Parameters
        ----------
        kernel_launch_info : dict
            Maps kernel func_var name -> (old_kernel, precompute_stmts, extra_launch_args)
        """
        super().__init__()
        self.kernel_launch_info = kernel_launch_info

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        kernel_name = stmt.func_var.name
        if kernel_name not in self.kernel_launch_info:
            return stmt
        old_kernel, precompute_stmts_template, extra_launch_args_template = self.kernel_launch_info[kernel_name]

        # The launch stmt args correspond to the old kernel's params.
        # Remap the precompute stmts (which reference kernel params) to use the launch args.
        param_remap: Dict[Var, Expr] = {kp: arg for kp, arg in zip(old_kernel.params, stmt.args)}

        # Create fresh vars and remap the precompute expressions for this call site
        precompute_stmts = []
        extra_launch_args = []
        for tmpl_stmt, tmpl_arg in zip(precompute_stmts_template, extra_launch_args_template):
            fresh_var = Var(tmpl_arg.name, type=tmpl_arg.type)
            precompute_stmts.append(DeclareStmt(fresh_var, init=rewrite(tmpl_stmt.init, param_remap)))
            extra_launch_args.append(fresh_var)

        new_launch = LaunchKernelStmt(
            func_var=stmt.func_var,
            args=list(stmt.args) + extra_launch_args,
            grid_dim=stmt.grid_dim,
            cluster_dim=stmt.cluster_dim,
            block_dim=stmt.block_dim,
            shared_mem=stmt.shared_mem_bytes,
            target=stmt.target,
        )
        return SeqStmt(precompute_stmts + [new_launch])


class LowerFastDivPass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        kernel_funcs: Dict[str, Function] = {}
        host_funcs: Dict[str, Function] = {}
        for name, func in ir_module.functions.items():
            if func.kind in ("cuda_kernel", "hip_kernel"):
                kernel_funcs[name] = func
            elif func.kind == "public":
                host_funcs[name] = func

        if len(kernel_funcs) == 0:
            return ir_module

        # Process each kernel independently
        new_funcs = dict(ir_module.functions)
        # Maps kernel func_var name -> (old_kernel, precompute_stmts, extra_launch_args)
        # Used later to rewrite all host functions in a single pass.
        kernel_launch_info: Dict[str, Tuple[Function, List, List[Var]]] = {}

        for kernel_name, kernel_func in kernel_funcs.items():
            tracker = GridConstantTracker(kernel_func.params)
            rewriter = FastDivRewriter(tracker)
            new_kernel_body = rewriter.visit(kernel_func.body)

            if not rewriter.found_fastdiv:
                continue

            # Build new kernel params
            new_kernel_params = list(kernel_func.params)
            divisor_entries = list(rewriter.divisor_map.values())
            for _, m_var, s_var in divisor_entries:
                new_kernel_params.append(m_var)
                new_kernel_params.append(s_var)

            new_kernel = Function(
                name=kernel_func.name,
                params=new_kernel_params,
                body=new_kernel_body,
                ret_type=kernel_func.ret_type,
                kind=kernel_func.kind,
                attrs=kernel_func.attrs,
            )
            new_funcs[kernel_name] = new_kernel

            # Build precompute stmts for host functions
            precompute_stmts, extra_launch_args = self._build_precompute(kernel_func, divisor_entries)
            kernel_launch_info[kernel_name] = (kernel_func, precompute_stmts, extra_launch_args)

        if not kernel_launch_info:
            return ir_module  # no fastdiv found in any kernel

        # Rewrite all host functions in one pass
        launch_rewriter = LaunchStmtRewriter(kernel_launch_info)
        for name, func in host_funcs.items():
            new_body = launch_rewriter.visit(func.body)
            if new_body is not func.body:
                new_funcs[name] = Function(
                    name=func.name,
                    params=func.params,
                    body=new_body,
                    ret_type=func.ret_type,
                    kind=func.kind,
                    attrs=func.attrs,
                )

        return ir_module.with_functions(new_funcs, ir_module.global_vars)

    @staticmethod
    def _build_precompute(
        old_kernel: Function,
        divisor_entries: List[Tuple[Expr, Var, Var]],
    ) -> Tuple[List, List[Var]]:
        """Build precompute DeclareStmts and extra launch args for a kernel's divisors.

        The precompute stmts use the old kernel's param variables. The LaunchStmtRewriter
        will place them before each LaunchKernelStmt, where the old kernel's params are
        in scope (the launch function's params mirror the kernel's params).
        """
        precompute_stmts = []
        extra_launch_args = []
        for divisor_expr, _, _ in divisor_entries:
            launch_m = Var("fast_div_m", type=int32)
            launch_s = Var("fast_div_s", type=int32)
            # Precompute functions return uint32; cast to int32 for the kernel params
            # to keep everything in int32 and avoid signed/unsigned casts that prevent
            # ptxas from using uniform registers.
            precompute_stmts.append(DeclareStmt(launch_m, init=Cast(fastdiv_precompute_m(divisor_expr), int32)))
            precompute_stmts.append(DeclareStmt(launch_s, init=Cast(fastdiv_precompute_s(divisor_expr), int32)))
            extra_launch_args.append(launch_m)
            extra_launch_args.append(launch_s)
        return precompute_stmts, extra_launch_args


def lower_fast_div_pass() -> Pass:
    return LowerFastDivPass()
