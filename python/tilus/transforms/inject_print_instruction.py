from typing import List, Optional, Type
from hidet.ir.dtypes import int32, boolean
from hidet.ir.expr import Expr, cast, logical_and

from tilus.extensions.hidet.ir.expr import as_expr
from tilus.ir.func import Function
from tilus.ir.builders import StmtBuilder
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
from tilus.ir.stmt import SeqStmt, ForStmt, InstructionStmt, seq_stmt, Stmt
from tilus.transforms.base import Pass

from hidet.ir.primitives import blockIdx


class InjectPrintInstructionRewriter(IRRewriter):
    def __init__(self, block_to_print: tuple[int, int, int], instructions_to_print: Optional[List[str]]):
        super().__init__()
        self.vm_printer = IRRewriter()
        self.block_to_print: tuple[int, int, int] = block_to_print
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

    def visit_Function(self, func: Function) -> Function:
        self.cond = logical_and(
            blockIdx.x == self.block_to_print[0],  # type: ignore[attr-defined]
            blockIdx.y == self.block_to_print[1],  # type: ignore[attr-defined]
            blockIdx.z == self.block_to_print[2],  # type: ignore[attr-defined]
        )

        prog_text = str(self.vm_printer(func))
        func = super().visit_Function(func)
        text = "Virtual Machine Program:\n{}\nPrint for {}\n".format(prog_text, str(self.block_to_print)).replace(
            "\n", "\\n"
        )
        new_body = SeqStmt(
            (
                InstructionStmt(FormatPrintInst.create(cond=self.cond, fstring="%s", expressions=[as_expr(text)])),
                func.body,
            )
        )
        return func.with_body(new_body)

    def visit_ForStmt(self, stmt: ForStmt) -> Stmt:
        vb = StmtBuilder()

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

    def visit_InstructionStmt(self, stmt: InstructionStmt) -> Stmt:
        inst: Instruction = self.visit(stmt.inst)

        if self.instructions_to_print and not isinstance(inst, tuple(self.instructions_to_print)):
            # specified the set of instructions to print, but the current instruction is not in the set
            return InstructionStmt(inst)

        assert isinstance(inst, Instruction)

        inst_string = "{}:\n".format(self.vm_printer(inst))

        # print the input of some instructions if they do not produce a tensor
        inst2input = {StoreGlobalInst: 0, StoreSharedInst: 0}
        skip_list = (ViewSharedInst,)

        if isinstance(inst, skip_list):
            return InstructionStmt(inst)

        if isinstance(inst, AllocateSharedInst) and inst.init is None:
            return InstructionStmt(inst)

        if isinstance(inst, AllocateInst) and inst.init is None:
            return InstructionStmt(inst)

        if inst.output is not None:
            from tilus.ir.inst import ElementwiseBinaryInst

            if isinstance(inst, ElementwiseBinaryInst):
                return seq_stmt(
                    [
                        inst,
                        PrintValueInst.create(inst.output, cond=self.cond, msg=inst_string),
                        FormatPrintInst.create(cond=self.cond, fstring="\n"),
                    ]
                )
            return seq_stmt(
                [
                    inst,
                    PrintValueInst.create(inst.output, cond=self.cond, msg=inst_string),
                    FormatPrintInst.create(cond=self.cond, fstring="\n"),
                ]
            )
        elif isinstance(inst, CopyAsyncInst):
            return seq_stmt(
                [
                    inst,
                    CopyAsyncWaitAllInst.create(),
                    PrintValueInst.create(inst.inputs[0], cond=self.cond, msg=inst_string),
                    FormatPrintInst.create(cond=self.cond, fstring="\n"),
                ]
            )
        elif type(inst) in inst2input:
            input_idx = inst2input[type(inst)]
            return seq_stmt(
                [
                    inst,
                    PrintValueInst.create(inst.inputs[input_idx], cond=self.cond, msg=inst_string),
                    FormatPrintInst.create(cond=self.cond, fstring="\n"),
                ]
            )
        else:
            return InstructionStmt(inst)


class InjectPrintInstructionPass(Pass):
    def __init__(self, block_to_print: tuple[int, int, int], instructions_to_print: Optional[List[str]]):
        super().__init__()
        self.block_to_print: tuple[int, int, int] = block_to_print
        self.instructions_to_print: Optional[List[str]] = instructions_to_print

    def __call__(self, prog: Function) -> Function:
        rewriter = InjectPrintInstructionRewriter(self.block_to_print, self.instructions_to_print)
        return rewriter(prog)


def inject_print_instruction_pass(
    block_to_print: tuple[int, int, int], instructions_to_print: Optional[List[str]]
) -> Pass:
    return InjectPrintInstructionPass(block_to_print=block_to_print, instructions_to_print=instructions_to_print)
