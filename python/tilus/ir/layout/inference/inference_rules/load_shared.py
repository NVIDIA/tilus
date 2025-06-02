from tilus import RegisterLayout, SharedLayout
from tilus.ir import RegisterTensor, SharedTensor
from tilus.ir.analyzers.grid_analyzer import analyze_grid
from tilus.ir.instructions import LoadMatrixInst, LoadSharedGenericInst, LoadSharedInst
from tilus.ir.instructions.cuda.ldmatrix import LoadMatrixConfig
from tilus.ir.layout import LayoutOperationError, rl_ops
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.utils import gcd


@register_rule(LoadMatrixInst)
@register_rule(LoadSharedInst)
@register_rule(LoadSharedGenericInst)
class LoadSharedInferSwizzledSharedRule(LayoutInferenceRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: LoadSharedInst | LoadSharedGenericInst | LoadMatrixInst
    ) -> dict[SharedTensor, SharedLayout]:
        a = inst.shared_input
        b = inst.register_output

        if not (a.optional_layout is None and b.optional_layout is not None):
            return {}

        for config in LoadMatrixConfig.all():
            if config.nbytes != a.dtype.nbytes:
                continue
            try:
                rl_ops.divide(b.layout, config.ldmatrix_layout)
            except LayoutOperationError:
                continue

            # use swizzle layout since we are using ldmatrix instruction
            from tilus.lang.modules.cuda import cuda

            return {a: cuda.swizzled_shared_layout(dtype=a.dtype, shape=a.shape)}

        return {}


@register_rule(LoadMatrixInst)
@register_rule(LoadSharedInst)
@register_rule(LoadSharedGenericInst)
class LoadSharedInferRegisterRule(LayoutInferenceRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: LoadSharedInst | LoadSharedGenericInst | LoadMatrixInst
    ) -> dict[RegisterTensor, RegisterLayout]:
        shared = inst.shared_input
        register = inst.register_output

        if not (shared.has_layout() and not register.has_layout()):
            return {}

        axes = shared.layout.axes
        offset = shared.layout.offset

        info = analyze_grid(
            shape=shared.shape,
            axes=axes,
            expr=offset,
            analysis=ctx.analysis,
        )

        for dim in range(len(shared.shape)):
            factor = gcd(
                info[dim].divisibility,
                info[dim].continuity,
                128 // shared.dtype.nbits,
                shared.shape[dim],
            )
            if factor > 1:
                lhs_shape = list(shared.shape)
                lhs_shape[dim] = shared.shape[dim] // factor
                rhs_shape = [1 if i != dim else factor for i in range(len(shared.shape))]
                layout = rl_ops.auto_repeat_spatial(num_threads=ctx.num_threads, shape=lhs_shape) * rl_ops.repeat(
                    *rhs_shape
                )
                return {register: layout}

        return {register: rl_ops.auto_repeat_spatial(num_threads=ctx.num_threads, shape=shared.shape)}
