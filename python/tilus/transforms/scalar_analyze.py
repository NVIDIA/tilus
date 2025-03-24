from tilus.ir.analyzers.scalar_analyzer import analyze_scalar
from tilus.ir.func import Function
from tilus.transforms.base import Pass


class AnalyzeScalarPass(Pass):
    def process_function(self, function: Function) -> Function:
        return analyze_scalar(function)


def analyze_scalar_pass() -> Pass:
    return AnalyzeScalarPass()
