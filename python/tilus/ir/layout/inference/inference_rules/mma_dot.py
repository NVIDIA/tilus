from tilus.ir.instructions import MmaDotInst
from tilus.ir.instructions.cuda.mma_dot import AtomicMmaConfig
from tilus.ir.layout import LayoutOperationError, RegisterLayout, divide
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(MmaDotInst)
class MmaDotRule(LayoutInferenceRule):
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

    @staticmethod
    def generate_default_layouts(
        num_warps: int, a: RegisterTensor, b: RegisterTensor, c: RegisterTensor, d: RegisterTensor
    ) -> dict[RegisterTensor, RegisterLayout]:
        from tilus.lang.modules.cuda import cuda

        assert len(a.shape) == len(b.shape) == len(c.shape) == len(d.shape) == 2, "MMA dot requires 2D tensors."

        m = a.shape[0]
        n = b.shape[1]
        k = a.shape[1]

        mma = cuda.default_dot_config(operand_dtype=a.dtype, acc_dtype=c.dtype, num_warps=num_warps, m=m, n=n, k=k)
        return {a: mma.la, b: mma.lb, c: mma.lc, d: mma.lc}

    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: MmaDotInst) -> dict[RegisterTensor, RegisterLayout]:
        a = inst.inputs[0].as_register_tensor()
        b = inst.inputs[1].as_register_tensor()
        c = inst.inputs[2].as_register_tensor()
        d = inst.output.as_register_tensor()

        if all(tensor.optional_layout is None for tensor in (a, b, c, d)):
            return MmaDotRule.generate_default_layouts(ctx.num_warps, a, b, c, d)
        else:
            return {}
