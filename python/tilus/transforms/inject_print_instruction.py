from typing import List, Dict, Optional, Type
import warnings
from hidet.ir.dtypes import int32, boolean
from hidet.ir.expr import Expr, cast, logical_and

from tilus.extensions.hidet.ir.expr import convert_to_expr
from tilus.ir.func import Function
from tilus.ir.builders import StatementBuilder
from tilus.ir.functors import IRRewriter
import tilus.ir.inst
from tilus.ir.inst import (
    Instruction,
    PrintValueInst,
    FormatPrintInst,
    CopyAsyncInst,
    CopyAsyncWaitAllInst,
    StoreGlobalInst,
    StoreSharedInst,
    ViewSharedInst,
    AllocateSharedInst,
    AllocateInst,
)
from tilus.ir.stmt import SeqStmt, ForStmt, InstructionStmt, seq_stmt
from tilus.transforms.base import Pass


class InjectPrintInstructionRewriter(IRRewriter):
    def __init__(self, block_to_print: Optional[Dict[str, int]], instructions_to_print: Optional[List[str]]):
        super().__init__()
        self.vm_printer = IRRewriter()
        self.block_to_print: Optional[Dict[str, int]] = block_to_print
        self.instructions_to_print: Optional[List[Type[Instruction]]] = None
        self.cond: Expr = boolean.true

        # check the existence of the instructions
        if instructions_to_print is not None:
            self.instructions_to_print = []
            for inst in instructions_to_print:
                if not hasattr(tilus.ir.inst, inst):
                    raise ValueError("Instruction {} does not exist".format(inst))
                self.instructions_to_print.append(getattr(tilus.ir.inst, inst))
        else:
            self.instructions_to_print = None

    def visit_Function(self, func: Function):
        self.cond = boolean.true
        block_to_print = self.block_to_print.copy() if self.block_to_print else {}
        block_printed = {}
        for axis in func.block_mapping.virtual_axes_values.keys():
            assert axis.hint is not None
            if block_to_print and axis.hint in block_to_print:
                val = int32(block_to_print[axis.hint])
                block_printed[axis.hint] = val
                del block_to_print[axis.hint]
            else:
                val = int32.zero
                block_printed[axis.hint] = 0
            self.cond = logical_and(self.cond, axis == val)
        if block_to_print:
            warnings.warn("Some block axes are specified but not used by the vm: {}".format(block_to_print))

        prog_text = str(self.vm_printer(func))
        func = super().visit_Function(func)
        text = "Virtual Machine Program:\n{}\nPrint for {}\n".format(prog_text, str(block_printed)).replace("\n", "\\n")
        func.body = SeqStmt(
            [
                InstructionStmt(FormatPrintInst(cond=self.cond, fstring="%s", expressions=[convert_to_expr(text)])),
                func.body,
            ]
        )
        return func

    def visit_ForStmt(self, stmt: ForStmt):
        vb = StatementBuilder()

        vb.format_print(
            fstring="for {} in range({}) when {} = %d:\n".format(
                self.vm_printer(stmt.iter_var), self.vm_printer(stmt.extent), self.vm_printer(stmt.iter_var)
            ),
            expressions=[cast(stmt.iter_var, int32)],
            cond=self.cond,
        )
        vb.append(self.visit(stmt.body))
        vb.format_print(
            fstring="end for {} in range({})\n\n".format(self.vm_printer(stmt.iter_var), self.vm_printer(stmt.extent)),
            expressions=[],
            cond=self.cond,
        )
        return ForStmt(
            iter_var=stmt.iter_var, extent=stmt.extent, body=vb.flush_statement(), unroll_factor=stmt.unroll_factor
        )

    def visit_Instruction(self, inst: Instruction):
        inst = super().visit_Instruction(inst)

        if self.instructions_to_print and not isinstance(inst, tuple(self.instructions_to_print)):
            # specified the set of instructions to print, but the current instruction is not in the set
            return inst

        inst_string = "{}:\n".format(self.vm_printer(inst))

        # print the input of some instructions if they do not produce a tensor
        inst2input = {StoreGlobalInst: 0, StoreSharedInst: 0}
        skip_list = (ViewSharedInst,)

        if isinstance(skip_list, skip_list):
            return inst

        if isinstance(inst, AllocateSharedInst) and inst.init is None:
            return inst

        if isinstance(inst, AllocateInst) and inst.init is None:
            return inst

        if inst.output is not None:
            from tilus.ir.inst import ElementwiseBinaryInst

            if isinstance(inst, ElementwiseBinaryInst):
                return seq_stmt(
                    [
                        # PrintValueInst(inst.inputs[0], cond=self.cond, msg=inst_string),
                        # PrintValueInst(inst.inputs[1], cond=self.cond, msg=''),
                        inst,
                        PrintValueInst(inst.output, cond=self.cond, msg=inst_string),
                        FormatPrintInst(cond=self.cond, fstring="\n"),
                    ]
                )
            return seq_stmt(
                [
                    inst,
                    PrintValueInst(inst.output, cond=self.cond, msg=inst_string),
                    FormatPrintInst(cond=self.cond, fstring="\n"),
                ]
            )
        elif isinstance(inst, CopyAsyncInst):
            return seq_stmt(
                [
                    inst,
                    CopyAsyncWaitAllInst(),
                    PrintValueInst(inst.inputs[0], cond=self.cond, msg=inst_string),
                    FormatPrintInst(cond=self.cond, fstring="\n"),
                ]
            )
        elif type(inst) in inst2input:
            input_idx = inst2input[type(inst)]
            return seq_stmt(
                [
                    inst,
                    PrintValueInst(inst.inputs[input_idx], cond=self.cond, msg=inst_string),
                    FormatPrintInst(cond=self.cond, fstring="\n"),
                ]
            )
        else:
            return inst


class InjectPrintInstructionPass(Pass):
    def __init__(self, block_to_print: Optional[Dict[str, int]], instructions_to_print: Optional[List[str]]):
        super().__init__()
        self.block_to_print: Optional[Dict[str, int]] = block_to_print
        self.instructions_to_print: Optional[List[str]] = instructions_to_print

    def __call__(self, prog: Function) -> Function:
        rewriter = InjectPrintInstructionRewriter(self.block_to_print, self.instructions_to_print)
        return rewriter(prog)


def inject_print_instruction_pass(
    block_to_print: Optional[Dict[str, int]], instructions_to_print: Optional[List[str]]
) -> Pass:
    return InjectPrintInstructionPass(block_to_print=block_to_print, instructions_to_print=instructions_to_print)
