from tilus import RegisterLayout
from tilus.ir.instructions import ReduceInst
from tilus.ir.layout import rl_ops
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(ReduceInst)
class ReduceRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: ReduceInst) -> dict[RegisterTensor, RegisterLayout]:
        x = inst.register_input
        y = inst.register_output

        if x.optional_layout is not None and y.optional_layout is not None:
            return {}
        elif x.optional_layout is not None:
            return {y: rl_ops.reduce(x.layout, dims=[inst.dim], keepdims=inst.keepdim)}
        elif y.optional_layout is not None:
            return {}
        else:
            return {}
