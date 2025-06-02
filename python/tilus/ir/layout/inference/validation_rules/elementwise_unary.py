from tilus.ir.instructions import CastInst, ElementwiseUnaryInst, Instruction
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(ElementwiseUnaryInst)
@register_rule(CastInst)
class UnaryRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: Instruction) -> bool:
        assert len(inst.inputs) == 1 and inst.output is not None
        assert isinstance(inst.inputs[0], RegisterTensor) and isinstance(inst.output, RegisterTensor)
        return inst.register_input.layout == inst.register_output.layout
