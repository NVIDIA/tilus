from tilus.ir.instructions import MmaDotInst
from tilus.ir.instructions.cuda.mma_dot import AtomicMmaConfig
from tilus.ir.layout import LayoutOperationError, divide
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule


@register_rule(MmaDotInst)
class MmaDotRule(LayoutValidationRule):
    """
    Layout inference rule for MMA dot instructions.
    """

    @staticmethod
    def validate(inst: MmaDotInst) -> bool:
        from tilus.ir.layout.mfunction_ops import identity

        a = inst.inputs[0].as_register_tensor()
        b = inst.inputs[1].as_register_tensor()
        c = inst.inputs[2].as_register_tensor()
        d = inst.output.as_register_tensor()
        for config in AtomicMmaConfig.all_configs().values():
            if not (a.dtype == b.dtype == config.operand_type and c.dtype == d.dtype == config.acc_type):
                continue

            try:
                outer_a = divide(a.layout, config.la)
                outer_b = divide(b.layout, config.lb)
                outer_c = divide(c.layout, config.lc)
                outer_d = divide(d.layout, config.lc)
            except LayoutOperationError:
                continue

            m, n, k = outer_d.shape[0], outer_d.shape[1], outer_a.shape[1]

            mf_g = identity([m, k, n])
            mf_a = mf_g.collapse(dims=[2]) * outer_a.spatial_mfunction()
            mf_b = mf_g.collapse(dims=[0]) * outer_b.spatial_mfunction()
            mf_c = mf_g.collapse(dims=[1]) * outer_c.spatial_mfunction()
            mf_d = mf_g.collapse(dims=[1]) * outer_d.spatial_mfunction()

            if any(not mf_operand.cover(mf_d) for mf_operand in (mf_a, mf_b, mf_c)):
                continue

            return True

        return False
