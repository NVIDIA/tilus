from tilus.ir.function import Function


class Pass:
    def __call__(self, prog: Function) -> Function:
        raise NotImplementedError()
