from tilus import RegisterLayout
from tilus.ir.instructions import TransposeInst
from tilus.ir.layout import rl_ops
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(TransposeInst)
class TransposeRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: TransposeInst) -> dict[RegisterTensor, RegisterLayout]:
        x = inst.register_input
        y = inst.register_output

        if x.optional_layout is not None and y.optional_layout is not None:
            return {}
        elif x.optional_layout is not None:
            return {y: rl_ops.permute(x.layout, dims=[1, 0])}
        elif y.optional_layout is not None:
            return {x: rl_ops.permute(y.layout, dims=[1, 0])}
        else:
            return {}
