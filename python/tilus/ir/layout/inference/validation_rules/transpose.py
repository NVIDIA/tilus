from tilus.ir.instructions import TransposeInst
from tilus.ir.layout import rl_ops
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(TransposeInst)
class TransposeRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: TransposeInst) -> bool:
        x: RegisterTensor = inst.register_input
        y: RegisterTensor = inst.register_output

        return x.layout == rl_ops.permute(y.layout, [1, 0])
