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
# """
# Dead code elimination pass.

# This pass eliminates unused scalar variables and unused register tensors.
# """

# from collections import defaultdict
# from typing import Dict, TypeAlias

# from hidet.ir.expr import Address, Expr, Reference, Var

# from tilus.ir.func import Function
# from tilus.ir.functors import IRRewriter, IRVisitor
# from tilus.ir.tensor import Tensor
# from tilus.ir.stmt import AssignStmt, DeclareStmt, LetStmt, SeqStmt, Stmt
# from tilus.ir.inst import Instruction
# from tilus.ir.tools import collect
# from tilus.transforms.base import Pass



# class UsageStatistics:
#     def __init__(self):
#         self.users: dict[Var | Tensor, list[Instruction | Stmt]] = {}
#         self.producer: dict[Var | Tensor, Instruction | Stmt] = {}

# class UsageAnalyzer(IRVisitor):
#     def __init__(self):
#         super().__init__()
#         self.stats = UsageStatistics()
    
#     def visit_Instruction(self, inst):
#         pass

#     def visit_DeclareStmt(self, stmt):
#         return super().visit_DeclareStmt(stmt)
    
#     def visit_LetStmt(self, stmt):
#         return super().visit_LetStmt(stmt)
    
#     def visit_AssignStmt(self, stmt):
#         return super().visit_AssignStmt(stmt)



# class DeclareToLetPass(Pass):
#     def process_function(self, func: Function) -> Function:
#         rewriter = DeclareToLetRewriter()
#         return rewriter.rewrite(func)


# def declare_to_let_pass() -> Pass:
#     return DeclareToLetPass()

