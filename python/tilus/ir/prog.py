from tilus.ir.func import Function


class Program:
    def __init__(self, functions: dict[str, Function]):
        self.functions: dict[str, Function] = functions

    def __str__(self):
        from tilus.ir.tools import IRPrinter

        printer = IRPrinter()
        return str(printer(self))
