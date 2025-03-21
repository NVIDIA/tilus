import functools
from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class IRNode:
    @functools.lru_cache(maxsize=1024)
    def __str__(self):
        from tilus.ir.tools import IRPrinter

        printer = IRPrinter()
        return str(printer(self))

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
