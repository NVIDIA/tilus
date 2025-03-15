from typing import List
from tilus.ir.func import Function
from tilus.ir.functors import IRVisitor
from tilus.ir.inst import Instruction


class InstructionCollector(IRVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.instructions: List[Instruction] = []

    def visit_Instruction(self, inst: Instruction):
        self.instructions.append(inst)


def collect_instructions(prog: Function) -> List[Instruction]:
    collector = InstructionCollector()
    collector(prog)
    return collector.instructions
