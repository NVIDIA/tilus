from dataclasses import dataclass, field
from typing import Type

from hidet.ir.dtypes import boolean, int32
from hidet.ir.expr import Expr, as_expr, cast, logical_and

from tilus.ir.builders import StmtBuilder
from tilus.ir.func import Function
from tilus.ir.functors import IRRewriter
from tilus.ir.inst import Instruction
from tilus.ir.instructions import (
    AllocateRegisterInst,
    AllocateSharedInst,
    CastInst,
    LoadGlobalGenericInst,
    LoadGlobalInst,
    LoadSharedInst,
    MmaDotInst,
    StoreGlobalInst,
)
from tilus.ir.stmt import ForStmt, Stmt
from tilus.transforms.base import Pass


@dataclass(frozen=True)
class PrintConfig:
    output: bool = False
    inputs: list[int] = field(default_factory=lambda: [])


PRINT_CONFIGS: dict[Type[Instruction], PrintConfig] = {
    AllocateRegisterInst: PrintConfig(output=True),
    AllocateSharedInst: PrintConfig(output=True),
    LoadGlobalInst: PrintConfig(output=True),
    LoadGlobalGenericInst: PrintConfig(output=True),
    LoadSharedInst: PrintConfig(output=True),
    MmaDotInst: PrintConfig(output=True),
    CastInst: PrintConfig(output=True),
    StoreGlobalInst: PrintConfig(inputs=[0]),
}


class InjectPrintInstructionRewriter(IRRewriter):
    def __init__(self, block_to_print: tuple[int, int, int]):
        super().__init__()
        self.vm_printer = IRRewriter()
        self.block_to_print: tuple[int, int, int] = block_to_print
        self.cond: Expr = boolean.true

    def visit_Function(self, func: Function) -> Function:
        self.cond = logical_and(*[a == b for a, b in zip(func.metadata.block_indices, self.block_to_print)])

        prog_text = str(self.vm_printer(func))
        func = super().visit_Function(func)
        text = "Virtual Machine Program:\n{}\nPrint for {}\n".format(prog_text, str(self.block_to_print)).replace(
            "\n", "\\n"
        )
        sb = StmtBuilder()
        sb.printf("%s\n", as_expr(text), cond=self.cond)
        sb.append(func.body)
        sb.printf(
            "end of block (%d, %d, %d)\n",
            self.block_to_print[0],
            self.block_to_print[1],
            self.block_to_print[2],
            cond=self.cond,
        )
        return func.with_body(sb.flush_stmts())

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
            iter_var=stmt.iter_var, extent=stmt.extent, body=vb.flush_stmts(), unroll_factor=stmt.unroll_factor
        )

    def visit_Instruction(self, inst: Instruction) -> Stmt | Instruction:
        inst = super().visit_Instruction(inst)

        if type(inst) not in PRINT_CONFIGS:
            # do not print the instruction
            return inst

        config = PRINT_CONFIGS[type(inst)]

        sb = StmtBuilder()
        sb.append(inst)
        sb.printf("%s\n", as_expr("{}".format(self.vm_printer(inst)).replace("\n", "\\n")), cond=self.cond)
        for input_idx in config.inputs:
            sb.print_tensor("input[0]: ", inst.inputs[input_idx].as_register_or_shared_tensor(), cond=self.cond)
        if config.output:
            sb.print_tensor("output: ", inst.register_or_shared_output, cond=self.cond)
        sb.printf("\n", cond=self.cond)
        return sb.flush_stmts()


class InjectPrintInstructionPass(Pass):
    def __init__(self, block_to_print: tuple[int, int, int]):
        super().__init__()
        self.block_to_print: tuple[int, int, int] = block_to_print

    def __call__(self, prog: Function) -> Function:
        rewriter = InjectPrintInstructionRewriter(self.block_to_print)
        return rewriter(prog)


def inject_print_instruction_pass(block_to_print: tuple[int, int, int]) -> Pass:
    return InjectPrintInstructionPass(block_to_print=block_to_print)
