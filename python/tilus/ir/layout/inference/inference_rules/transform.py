from tilus import RegisterLayout
from tilus.ir.instructions import SqueezeInst, UnsqueezeInst
from tilus.ir.layout import ops
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(UnsqueezeInst)
class UnsqueezeRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: UnsqueezeInst) -> dict[RegisterTensor, RegisterLayout]:
        x = inst.register_input
        y = inst.register_output

        if x.optional_layout is not None and y.optional_layout is not None:
            return {}
        elif x.optional_layout is not None:
            return {y: ops.unsqueeze(x.layout, dims=inst.dims)}
        elif y.optional_layout is not None:
            return {x: ops.squeeze(y.layout, dims=inst.dims)}
        else:
            return {}


@register_rule(SqueezeInst)
class SqueezeRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: SqueezeInst) -> dict[RegisterTensor, RegisterLayout]:
        x = inst.register_input
        y = inst.register_output

        if x.optional_layout is not None and y.optional_layout is not None:
            return {}
        elif x.optional_layout is not None:
            return {y: ops.squeeze(x.layout, dims=inst.dims)}
        elif y.optional_layout is not None:
            return {x: ops.unsqueeze(y.layout, dims=inst.dims)}
        else:
            return {}
