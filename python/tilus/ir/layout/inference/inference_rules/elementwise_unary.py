from tilus import RegisterLayout
from tilus.ir.instructions import CastInst, ElementwiseUnaryInst, Instruction
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(ElementwiseUnaryInst)
@register_rule(CastInst)
class UnaryRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: Instruction) -> dict[RegisterTensor, RegisterLayout]:
        x = inst.register_input
        y = inst.register_output

        if x.optional_layout is not None and y.optional_layout is not None:
            return {}
        elif x.optional_layout is not None:
            return {y: x.layout}
        elif y.optional_layout is not None:
            return {x: y.layout}
        else:
            return {}
