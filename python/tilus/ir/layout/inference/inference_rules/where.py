from hidet.utils import same_list

from tilus import RegisterLayout
from tilus.ir.instructions import WhereInst
from tilus.ir.layout import ops
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(WhereInst)
class WhereRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: WhereInst) -> dict[RegisterTensor, RegisterLayout]:
        cond = inst.inputs[0].as_register_tensor()
        x = inst.inputs[1].as_register_tensor()
        y = inst.inputs[2].as_register_tensor()
        out = inst.register_output

        if out.optional_layout is not None:
            ret = {}
            for operand in (cond, x, y):
                if operand.optional_layout is None:
                    ret[operand] = ops.reduce_to(out.layout, shape=operand.shape)
            return ret
        else:
            for operand in (cond, x, y):
                if operand.optional_layout is not None and same_list(operand.shape, out.shape):
                    return {out: operand.layout}
        return {}
