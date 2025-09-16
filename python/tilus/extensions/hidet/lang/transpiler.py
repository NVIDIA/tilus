from hidet.lang.transpiler import PythonToHidetTranslator
import typing
import ast


class ExtendedPythonToHidetTranslator(PythonToHidetTranslator):
    def visit_Sequence(self, expr: ast.Tuple | ast.List) -> list:
        seq = []
        for v in expr.elts:
            if isinstance(v, ast.Starred):
                value = self.visit(v.value)
                if not isinstance(value, typing.Sequence):
                    raise ValueError(f'Expect a sequence after * in {ast.dump(expr)}, but got {value}')
                seq.extend(value)
            else:
                seq.append(self.visit(v))
        return seq

    def visit_Tuple(self, expr: ast.Tuple):
        return tuple(self.visit_Sequence(expr))

    def visit_List(self, expr: ast.List):
        return self.visit_Sequence(expr)
