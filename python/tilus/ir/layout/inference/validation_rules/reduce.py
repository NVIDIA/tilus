from tilus.ir.instructions import ReduceInst
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule


@register_rule(ReduceInst)
class ReduceRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: ReduceInst) -> bool:
        return True
