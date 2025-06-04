from tilus.ir.instructions import ReduceInst
from tilus.ir.layout import rl_ops
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule


@register_rule(ReduceInst)
class ReduceRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: ReduceInst) -> bool:
        src = inst.register_input
        dst = inst.register_output

        dst_layout = dst.layout

        if not inst.keepdim:
            dst_layout = rl_ops.unsqueeze(dst_layout, dims=[inst.dim])

        return src.layout.reduce_to(dst_layout.shape) == dst_layout
