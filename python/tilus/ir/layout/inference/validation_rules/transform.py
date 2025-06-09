from tilus.ir.instructions import SqueezeInst, UnsqueezeInst
from tilus.ir.layout import rl_ops
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(SqueezeInst)
class TransposeRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: SqueezeInst) -> bool:
        x: RegisterTensor = inst.register_input
        y: RegisterTensor = inst.register_output

        return y.layout == rl_ops.squeeze(x.layout, dims=inst.dims)


@register_rule(UnsqueezeInst)
class UnsqueezeRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: UnsqueezeInst) -> bool:
        x: RegisterTensor = inst.register_input
        y: RegisterTensor = inst.register_output

        return y.layout == rl_ops.unsqueeze(x.layout, dims=inst.dims)
