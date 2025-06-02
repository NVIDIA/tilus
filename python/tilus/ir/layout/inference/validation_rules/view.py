from tilus.ir.instructions import ViewInst
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(ViewInst)
class ViewRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: ViewInst) -> bool:
        x: RegisterTensor = inst.register_input
        y: RegisterTensor = inst.register_output

        if x.layout.local_size * x.dtype.nbits != y.layout.local_size * y.dtype.nbits:
            return False
        if x.layout.spatial_size != y.layout.spatial_size:
            return False
        return True
