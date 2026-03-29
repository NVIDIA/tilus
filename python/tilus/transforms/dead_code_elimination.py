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
"""
Dead code elimination pass for Tilus IR.

Removes functional instructions whose output tensors are never consumed by any other instruction.
"""

from typing import Type

from tilus.hidet.ir.expr import Expr, Var
from tilus.hidet.ir.tools import collect as hidet_collect
from tilus.ir.func import Function
from tilus.ir.functors import IRRewriter, IRVisitor
from tilus.ir.inst import Instruction
from tilus.ir.instructions.cuda.clc import ClusterLaunchControlQueryResponseInst
from tilus.ir.instructions.cuda.mapa import MapSharedAddrInst
from tilus.ir.instructions.cuda.mbarrier import AllocBarrierInst
from tilus.ir.instructions.cuda.mma_dot import DotInst
from tilus.ir.instructions.cuda.simt_dot import SimtDotInst
from tilus.ir.instructions.cuda.tcgen05 import Tcgen05LoadInst, Tcgen05SliceInst, Tcgen05ViewInst
from tilus.ir.instructions.generic import (
    AllocateRegisterInst,
    CastInst,
    ElementwiseBinaryBaseInst,
    ElementwiseUnaryBaseInst,
    GlobalViewInst,
    LoadGlobalGenericInst,
    LoadGlobalInst,
    LoadSharedInst,
    PermuteSharedInst,
    ReduceInst,
    RepeatInst,
    RepeatInterleaveInst,
    ReshapeSharedInst,
    SliceGlobalInst,
    SliceRegisterInst,
    SliceSharedInst,
    SqueezeInst,
    TransposeInst,
    UnsqueezeInst,
    ViewInst,
    WhereInst,
)
from tilus.ir.stmt import SeqStmt, Stmt, TensorItemPtrStmt, TensorItemValueStmt
from tilus.ir.tensor import Tensor
from tilus.transforms.base import Pass

# Functional instruction types: pure computations safe to eliminate if output is unused.
FUNCTIONAL_INST_TYPES: tuple[Type[Instruction], ...] = (
    # Register tensor operations
    AllocateRegisterInst,
    SliceRegisterInst,
    CastInst,
    ElementwiseUnaryBaseInst,  # covers NegInst, AbsInst, ClipInst, ElementwiseUnaryInst
    ElementwiseBinaryBaseInst,  # covers AddInst, SubInst, MulInst, DivInst, ModInst, ElementwiseBinaryInst
    WhereInst,
    RepeatInst,
    RepeatInterleaveInst,
    ReduceInst,
    ViewInst,
    SqueezeInst,
    UnsqueezeInst,
    TransposeInst,
    # Load instructions
    LoadGlobalInst,
    LoadSharedInst,
    LoadGlobalGenericInst,
    Tcgen05LoadInst,
    # Shared/Global tensor views
    SliceGlobalInst,
    SliceSharedInst,
    ReshapeSharedInst,
    PermuteSharedInst,
    GlobalViewInst,
    # TMemory views
    Tcgen05SliceInst,
    Tcgen05ViewInst,
    # Other pure ops
    DotInst,
    SimtDotInst,
    MapSharedAddrInst,
    ClusterLaunchControlQueryResponseInst,
    AllocBarrierInst,
)


def _is_functional(inst: Instruction) -> bool:
    return isinstance(inst, FUNCTIONAL_INST_TYPES)


class UsedTensorCollector(IRVisitor):
    """
    Collects all tensors that are "used" (consumed by a live instruction or statement).

    An instruction's input tensors are used if:
    - The instruction is side-effecting (not functional), OR
    - The instruction is functional and its output tensor is itself used.

    A TensorItemValueStmt/TensorItemPtrStmt's tensor is used only if the Var it
    binds is referenced in some expression elsewhere in the function.

    We iterate to a fixed point since a functional instruction's liveness
    depends on whether its output is consumed by another live instruction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.used_tensors: set[int] = set()  # set of id(tensor)
        self.functional_insts: list[Instruction] = []
        # Deferred: TensorItem stmts whose liveness depends on Var usage
        self.tensor_item_stmts: list[TensorItemValueStmt | TensorItemPtrStmt] = []
        # All Vars referenced in expressions (collected after traversal).
        # We skip visiting the defining Var in visit_TensorItemValueStmt/PtrStmt,
        # so a Var only appears here if it's referenced elsewhere.
        self.expr_vars: set[int] = set()  # set of id(Var)

    def visit_Instruction(self, inst: Instruction) -> None:
        if _is_functional(inst):
            self.functional_insts.append(inst)
        else:
            # Side-effecting: all inputs are unconditionally used
            for tensor in inst.inputs:
                self.used_tensors.add(id(tensor))
        # Collect Vars from all Expr-typed attributes so that TensorItemValueStmt
        # vars referenced in instruction attributes are tracked.
        for value in inst.attributes.values():
            self._collect_expr_vars(value)

    def _collect_expr_vars(self, value: Expr | tuple | list | object) -> None:
        """Recursively collect Vars from Expr-typed values (including inside tuples/lists)."""
        if isinstance(value, Expr):
            for var in hidet_collect(value, Var):
                self.expr_vars.add(id(var))
        elif isinstance(value, (tuple, list)):
            for item in value:
                self._collect_expr_vars(item)

    def visit_TensorItemValueStmt(self, stmt: TensorItemValueStmt) -> None:
        # Defer: only mark tensor as used if stmt.var is actually referenced.
        # We skip visiting stmt.var here so it won't self-register in expr_vars.
        self.tensor_item_stmts.append(stmt)
        self.visit(stmt.tensor)

    def visit_TensorItemPtrStmt(self, stmt: TensorItemPtrStmt) -> None:
        # Defer: only mark tensor as used if stmt.ptr_var is actually referenced.
        self.tensor_item_stmts.append(stmt)
        self.visit(stmt.tensor)

    def visit_Expr(self, expr: Expr) -> None:
        # Collect all Vars referenced in Hidet expressions.
        for var in hidet_collect(expr, Var):
            self.expr_vars.add(id(var))

    def _mark_used(self, tensor: Tensor) -> bool:
        """Mark a tensor as used. Returns True if it was newly added."""
        tid = id(tensor)
        if tid not in self.used_tensors:
            self.used_tensors.add(tid)
            return True
        return False

    def propagate(self) -> None:
        """Fixed-point propagation of tensor liveness."""
        # Mark tensors from TensorItem stmts whose bound Var is referenced
        for stmt in self.tensor_item_stmts:
            bound_var = stmt.var if isinstance(stmt, TensorItemValueStmt) else stmt.ptr_var
            if id(bound_var) in self.expr_vars:
                self.used_tensors.add(id(stmt.tensor))

        # Propagate through functional instruction chains
        changed = True
        while changed:
            changed = False
            for inst in self.functional_insts:
                if inst.output is not None and id(inst.output) in self.used_tensors:
                    for tensor in inst.inputs:
                        if self._mark_used(tensor):
                            changed = True


class DeadCodeEliminator(IRRewriter):
    """Eliminates dead functional instructions and dead TensorItem stmts."""

    def __init__(self, used_tensors: set[int]) -> None:
        super().__init__()
        self.used_tensors = used_tensors

    def visit_Instruction(self, inst: Instruction) -> Instruction | None:
        if _is_functional(inst) and inst.output is not None and id(inst.output) not in self.used_tensors:
            return None
        return super().visit_Instruction(inst)

    def visit_TensorItemValueStmt(self, stmt: TensorItemValueStmt) -> Stmt:
        if id(stmt.tensor) not in self.used_tensors:
            return SeqStmt(())
        return super().visit_TensorItemValueStmt(stmt)

    def visit_TensorItemPtrStmt(self, stmt: TensorItemPtrStmt) -> Stmt:
        if id(stmt.tensor) not in self.used_tensors:
            return SeqStmt(())
        return super().visit_TensorItemPtrStmt(stmt)


class DeadCodeEliminationPass(Pass):
    def process_function(self, function: Function) -> Function:
        # Pass 1: collect used tensors
        collector = UsedTensorCollector()
        collector.visit(function)
        collector.propagate()

        # Check if there's anything to eliminate
        has_dead = any(
            inst.output is not None and id(inst.output) not in collector.used_tensors
            for inst in collector.functional_insts
        ) or any(id(stmt.tensor) not in collector.used_tensors for stmt in collector.tensor_item_stmts)

        if not has_dead:
            return function

        # Pass 2: eliminate dead instructions
        eliminator = DeadCodeEliminator(collector.used_tensors)
        return eliminator.visit(function)


def dead_code_elimination_pass() -> Pass:
    return DeadCodeEliminationPass()
