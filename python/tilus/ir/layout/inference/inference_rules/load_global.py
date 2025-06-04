from hidet.ir.expr import Expr, Var, logical_and

from tilus import RegisterLayout
from tilus.extensions.hidet.ir.expr import index_vars
from tilus.ir.analyzers.grid_analyzer import TensorInfo, analyze_grid
from tilus.ir.instructions import LoadGlobalGenericInst, LoadGlobalInst
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.layout.register_layout_ops import auto_repeat_spatial
from tilus.ir.tensor import RegisterTensor
from tilus.utils import gcd, prod


@register_rule(LoadGlobalGenericInst)
@register_rule(LoadGlobalInst)
class LoadGlobalRule(LayoutInferenceRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: LoadGlobalInst | LoadGlobalGenericInst
    ) -> dict[RegisterTensor, RegisterLayout]:
        output = inst.register_output

        if output.optional_layout:
            # the layout has been determined, skip inference
            return {}

        # grid analysis over the offset and mask of each position in the output grid
        shape = output.shape
        axes: list[Var]
        offset: Expr
        mask: Expr
        if isinstance(inst, LoadGlobalGenericInst):
            axes = list(inst.axes)
            offset = inst.offset
            mask = inst.mask
        else:
            input_shape = inst.inputs[0].as_global_tensor().shape
            axes = index_vars(len(output.shape))
            global_offsets: list[Expr] = list(inst.offsets)
            for dim, axis in zip(inst.slice_dims, axes):
                global_offsets[dim] = global_offsets[dim] + axis
            offset = inst.inputs[0].as_global_tensor().layout(*global_offsets)
            mask = logical_and(*[logical_and(0 <= i, i < input_shape[dim]) for dim, i in enumerate(global_offsets)])
        offset_info: TensorInfo = analyze_grid(shape=shape, axes=axes, expr=offset, analysis=ctx.analysis)
        mask_info: TensorInfo = analyze_grid(shape=shape, axes=axes, expr=mask, analysis=ctx.analysis)

        # find the dimension to perform the vectorization and the vectorization factor
        dtype = inst.register_output.dtype
        max_factor = max(prod(shape) // ctx.num_threads, 1)
        for dim in range(len(shape)):
            factor = gcd(
                offset_info[dim].divisibility,
                offset_info[dim].continuity,
                mask_info[dim].constancy,
                128 // dtype.nbits,
                shape[dim],
                max_factor,
            )
            if factor > 1:
                lhs_shape = list(shape)
                lhs_shape[dim] = shape[dim] // factor
                rhs_shape = [1 if i != dim else factor for i in range(len(shape))]
                return {output: auto_repeat_spatial(num_threads=ctx.num_threads, shape=lhs_shape).repeat(*rhs_shape)}

        # fall back to a default layout
        return {output: auto_repeat_spatial(num_threads=ctx.num_threads, shape=shape)}
