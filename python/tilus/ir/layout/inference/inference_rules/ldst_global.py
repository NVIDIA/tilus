from hidet.ir.expr import Expr, Var, logical_and

from tilus import RegisterLayout
from tilus.extensions.hidet.ir.expr import index_vars
from tilus.ir.analyzers.grid_analyzer import TensorInfo, analyze_grid
from tilus.ir.instructions import LoadGlobalGenericInst, LoadGlobalInst, StoreGlobalGenericInst, StoreGlobalInst
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.layout.register_layout_ops import auto_repeat_spatial
from tilus.ir.tensor import RegisterTensor
from tilus.utils import gcd, prod


class LoadStoreGlobalRule(LayoutInferenceRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext,
        inst: StoreGlobalInst | StoreGlobalGenericInst | LoadGlobalInst | LoadGlobalGenericInst,
    ) -> dict[RegisterTensor, RegisterLayout]:
        if isinstance(inst, StoreGlobalInst):
            tensor = inst.inputs[1].as_register_tensor()
        elif isinstance(inst, StoreGlobalGenericInst):
            tensor = inst.inputs[0].as_register_tensor()
        elif isinstance(inst, (LoadGlobalInst, LoadGlobalGenericInst)):
            tensor = inst.register_output
        else:
            raise NotImplementedError(inst)

        if tensor.optional_layout:
            return {}

        # grid analysis over the offset and mask of each position in the output grid
        num_threads = ctx.num_threads
        analysis = ctx.analysis
        dtype = tensor.dtype
        shape = tensor.shape
        axes: list[Var]
        offset: Expr
        mask: Expr
        if isinstance(inst, (StoreGlobalGenericInst, LoadGlobalGenericInst)):
            axes = list(inst.axes)
            offset = inst.offset
            mask = inst.mask
        elif isinstance(inst, (LoadGlobalInst, StoreGlobalInst)):
            global_shape = inst.inputs[0].as_global_tensor().shape
            global_offsets: list[Expr] = list(inst.offsets)
            axes = index_vars(len(tensor.shape))
            for dim, axis in zip(inst.slice_dims, axes):
                global_offsets[dim] = global_offsets[dim] + axis
            offset = inst.inputs[0].as_global_tensor().layout(*global_offsets)
            mask = logical_and(*[logical_and(0 <= i, i < global_shape[dim]) for dim, i in enumerate(global_offsets)])
        else:
            assert False

        # analyze the offset and mask expressions to get the tensor information
        offset_info: TensorInfo = analyze_grid(shape=shape, axes=axes, expr=offset, analysis=analysis)
        mask_info: TensorInfo = analyze_grid(shape=shape, axes=axes, expr=mask, analysis=analysis)

        # find the dimension to perform the vectorization and the vectorization factor
        max_factor = max(prod(shape) // num_threads, 1)
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
                return {tensor: auto_repeat_spatial(num_threads=num_threads, shape=lhs_shape).repeat(*rhs_shape)}

        # fall back to a default layout
        return {tensor: auto_repeat_spatial(num_threads=num_threads, shape=shape)}


@register_rule(LoadGlobalGenericInst)
@register_rule(LoadGlobalInst)
class LoadGlobalRule(LoadStoreGlobalRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: LoadGlobalInst | LoadGlobalGenericInst
    ) -> dict[RegisterTensor, RegisterLayout]:
        return LoadStoreGlobalRule.inference(ctx, inst)


@register_rule(StoreGlobalInst)
@register_rule(StoreGlobalGenericInst)
class StoreGlobalRule(LoadStoreGlobalRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: StoreGlobalInst | StoreGlobalGenericInst
    ) -> dict[RegisterTensor, RegisterLayout]:
        return LoadStoreGlobalRule.inference(ctx, inst)
