from hidet.ir.expr import default_float_dtype
from hidet.ir.tools.type_infer import TypeInfer

from tilus.extensions import update


class UpdatedTypeInfer(TypeInfer):
    def visit_PyConstant(self, c):
        if isinstance(c, float):
            return default_float_dtype
        else:
            return super().visit_PyConstant(c)


@update("hidet.ir.tools.infer_type")
@update("hidet.ir.tools.type_infer.infer_type")
def infer_type(expr):
    infer = UpdatedTypeInfer()
    return infer(expr)
