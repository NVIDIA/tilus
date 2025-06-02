from hidet.utils import same_list

from tilus.ir.instructions import AddInst, DivInst, ElementwiseBinaryInst, Instruction, ModInst, MulInst, SubInst
from tilus.ir.layout import RegisterLayout
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(AddInst)
@register_rule(SubInst)
@register_rule(DivInst)
@register_rule(MulInst)
@register_rule(ModInst)
@register_rule(ElementwiseBinaryInst)
class BinaryRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: Instruction) -> dict[RegisterTensor, RegisterLayout]:
        a = inst.inputs[0].as_register_tensor()
        b = inst.inputs[1].as_register_tensor()
        c = inst.output.as_register_tensor()

        if all(tensor.optional_layout is None for tensor in (a, b, c)):
            return {}
        elif c.optional_layout is not None:
            # c => a | b
            mapping = {}
            if a.optional_layout is not None:
                mapping[a] = c.layout.reduce_to(a.shape)
            if b.optional_layout is not None:
                mapping[b] = c.layout.reduce_to(b.shape)
            return mapping
        elif a.optional_layout is not None and b.optional_layout is not None:
            if same_list(a.shape, c.shape):
                return {b: a.layout.reduce_to(b.shape), c: a.layout}
            elif same_list(b.shape, c.shape):
                return {a: b.layout.reduce_to(a.shape), c: b.layout}
            else:
                return {}
        else:
            return {}
