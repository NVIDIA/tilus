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
from typing import Union

from tilus.ir.builders import StmtBuilder
from tilus.ir.func import Function
from tilus.ir.functors import IRRewriter
from tilus.ir.inst import Instruction
from tilus.ir.instructions import PrintTensorInst
from tilus.ir.stmt import Stmt
from tilus.ir.tensor import TMemoryTensor
from tilus.transforms.base import Pass


class LowerPrintTMemoryTensorRewriter(IRRewriter):
    def visit_PrintTensorInst(self, inst: PrintTensorInst) -> Union[Instruction, Stmt]:
        if not isinstance(inst.inputs[0], TMemoryTensor):
            return super().visit_Instruction(inst)

        sb = StmtBuilder()

        tmem_tensor = inst.inputs[0].as_tmemory_tensor()
        regs_tensor = sb.tcgen05_load(tmem_tensor, offsets=[0, 0], shape=tmem_tensor.shape)
        sb.tcgen05_wait_load()
        sb.print_tensor(inst.msg, regs_tensor)
        return sb.flush_stmts()


class LowerPrintTMemoryTensorPass(Pass):
    def process_function(self, function: Function) -> Function:
        rewriter = LowerPrintTMemoryTensorRewriter()
        return rewriter.visit(function)


def lower_print_tmemory_tensor_pass() -> Pass:
    return LowerPrintTMemoryTensorPass()
