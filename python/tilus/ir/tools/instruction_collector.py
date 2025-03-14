from typing import List
from tilus.ir.program import VirtualMachineProgram
from tilus.ir.functor import VirtualMachineVisitor
from tilus.ir.inst import Instruction


class InstructionCollector(VirtualMachineVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.instructions: List[Instruction] = []

    def visit_Instruction(self, inst: Instruction):
        self.instructions.append(inst)


def collect_instructions(prog: VirtualMachineProgram) -> List[Instruction]:
    collector = InstructionCollector()
    collector(prog)
    return collector.instructions
