from tilus import SharedLayout
from tilus.ir import SharedTensor
from tilus.ir.instructions import StoreSharedGenericInst, StoreSharedInst
from tilus.ir.instructions.cuda.ldmatrix import LoadMatrixConfig
from tilus.ir.layout import LayoutOperationError, ops
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule


@register_rule(StoreSharedGenericInst)
@register_rule(StoreSharedInst)
class StoreSharedSwizzleRule(LayoutInferenceRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: StoreSharedInst | StoreSharedGenericInst
    ) -> dict[SharedTensor, SharedLayout]:
        a = inst.inputs[0].as_shared_tensor()
        b = inst.inputs[1].as_register_tensor()

        if not (a.optional_layout is None and b.optional_layout is not None):
            return {}

        for config in LoadMatrixConfig.all():
            if config.nbytes != a.dtype.nbytes:
                continue
            try:
                ops.divide(b.layout, config.ldmatrix_layout)
            except LayoutOperationError:
                continue

            # use swizzle layout since we are using ldmatrix instruction
            from tilus.lang.modules.cuda import cuda

            return {a: cuda.swizzled_shared_layout(dtype=a.dtype, shape=a.shape)}

        return {}
