from tilus import SharedLayout
from tilus.ir import SharedTensor
from tilus.ir.instructions import SharedSliceInst
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.layout.shared_layout import shared_compose, shared_repeat


@register_rule(SharedSliceInst)
class SharedSliceRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: SharedSliceInst) -> dict[SharedTensor, SharedLayout]:
        a = inst.shared_input
        b = inst.shared_output
        if a.optional_layout is not None and b.optional_layout is not None:
            return {}
        elif a.optional_layout is not None:
            return {b: a.layout.slice(offsets=inst.offsets, slice_dims=inst.dims, slice_shape=b.shape)}
        elif b.optional_layout is not None:
            b_layout = b.layout.unsqueeze(dims=range(len(a.shape) - len(b.shape)))
            outer_shape = []
            for i in range(len(a.shape)):
                outer_shape.append(a.shape[i] // b_layout.shape[i])
            return {a: shared_compose(shared_repeat(*outer_shape), b_layout).simplify()}
        else:
            return {}
