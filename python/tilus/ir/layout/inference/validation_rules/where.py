from tilus.ir.instructions import WhereInst
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule
from tilus.ir.mfunction import ops
from tilus.ir.tensor import RegisterTensor


@register_rule(WhereInst)
class WhereRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: WhereInst) -> bool:
        cond: RegisterTensor = inst.inputs[0].as_register_tensor()
        x: RegisterTensor = inst.inputs[1].as_register_tensor()
        y: RegisterTensor = inst.inputs[2].as_register_tensor()

        for operand in [cond, x, y]:
            out: RegisterTensor = inst.register_output

            fa = ops.identity(out.shape).collapse_by_shape(operand.shape) * operand.layout.spatial_mfunction()
            fb = out.layout.spatial_mfunction()

            if not fa.cover(fb):
                return False
        return True
