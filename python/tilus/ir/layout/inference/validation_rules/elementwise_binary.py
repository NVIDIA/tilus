from tilus.ir.instructions import AddInst, DivInst, ElementwiseBinaryInst, Instruction, ModInst, MulInst, SubInst
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule
from tilus.ir.mfunction import ops
from tilus.ir.tensor import RegisterTensor


@register_rule(AddInst)
@register_rule(SubInst)
@register_rule(DivInst)
@register_rule(MulInst)
@register_rule(ModInst)
@register_rule(ElementwiseBinaryInst)
class BinaryRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: Instruction) -> bool:
        assert len(inst.inputs) == 2 and inst.output is not None
        assert all(isinstance(tensor, RegisterTensor) for tensor in inst.inputs + (inst.output,))
        for i in range(2):
            x: RegisterTensor = inst.inputs[i].as_register_tensor()
            y: RegisterTensor = inst.register_output

            fa = ops.identity(y.shape).collapse_by_shape(x.shape) * x.layout.spatial_mfunction()
            fb = y.layout.spatial_mfunction()

            if not fa.cover(fb):
                return False
        return True
