from dataclasses import dataclass
from tilus.ir.func import Function
from tilus.ir.utils import frozendict
from tilus.ir.node import IRNode


@dataclass(frozen=True, eq=False)
class Program(IRNode):
    functions: frozendict[str, Function]

    @staticmethod
    def create(functions: dict[str, Function]):
        return Program(frozendict(functions))

    def with_function(self, new_function: Function):
        new_functions = dict(self.functions)
        new_functions[new_function.name] = new_function
        return Program(frozendict(new_functions))
