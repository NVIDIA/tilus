from tilus.ir.instructions import AssignInst
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(AssignInst)
class AssignRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: AssignInst) -> bool:
        x: RegisterTensor = inst.register_input
        y: RegisterTensor = inst.register_output

        fa = x.layout.spatial_mfunction()
        fb = y.layout.spatial_mfunction()

        return fa.cover(fb)
