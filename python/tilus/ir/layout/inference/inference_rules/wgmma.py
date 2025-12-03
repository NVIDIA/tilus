from tilus.ir.instructions.cuda.tcgen05 import Tcgen05MmaSSInst, Tcgen05MmaTSInst
from tilus.ir.layout import SharedLayout
from tilus.ir.layout.cuda.tcgen05.smem import (
    Tcgen05SwizzleMode,
    generate_canonical_layout,
)
from tilus.ir.layout.inference.rule import (
    LayoutInferenceContext,
    LayoutInferenceError,
    LayoutInferenceRule,
    register_rule,
)
from tilus.ir.tensor import SharedTensor, RegisterTensor
from tilus.ir.layout import RegisterLayout

from tilus.ir.instructions.cuda.wgmma import WgmmaMmaSSInst, WgmmaMmaRSInst

from tilus.ir.layout.ops.register_ops import spatial, local, column_spatial, column_local


def generate_wgmma_register_layout(num_column, dtype) -> RegisterLayout:
    T = 64 // dtype.nbits
    return column_spatial(4, 1).column_local(2, num_column // T // 4).spatial(8, 4).local(T)

@register_rule(WgmmaMmaSSInst)
class WgmmaMmaSSRule(LayoutInferenceRule):
    @staticmethod
    def inference(ctx: LayoutInferenceContext, inst: WgmmaMmaSSInst) -> dict[SharedTensor, SharedLayout]:
        a_tensor: SharedTensor = inst.inputs[0].as_shared_tensor()
        b_tensor: SharedTensor = inst.inputs[1].as_shared_tensor()
        d_tensor: RegisterTensor = inst.inputs[2].as_register_tensor()

        a_shape = a_tensor.shape
        b_shape = b_tensor.shape
        d_shape = d_tensor.shape

        if not len(a_shape) == len(b_shape) == len(d_shape) == 2:
            raise LayoutInferenceError(
                f"A, B, and D must have 2 dimensions, but got {len(a_shape)}, {len(b_shape)}, and {len(d_shape)}."
            )
        if a_shape[1] != b_shape[0] or a_shape[0] != d_shape[0] or b_shape[1] != d_shape[1]:
            raise LayoutInferenceError(
                f"A, B, and D must have compatible shapes, but got {a_tensor.shape}, {b_tensor.shape}, and {d_tensor.shape}."
            )
        m, n, k = d_shape[0], d_shape[1], a_shape[1]

        ret = {}
        if not a_tensor.has_layout():
            for swizzle_mode in [
                Tcgen05SwizzleMode.B128_SWIZZLE,
                Tcgen05SwizzleMode.B64_SWIZZLE,
                Tcgen05SwizzleMode.B32_SWIZZLE,
                Tcgen05SwizzleMode.NO_SWIZZLE,
            ]:
                try:
                    a_layout_canonical = generate_canonical_layout(
                        shape=(m, k), dtype=a_tensor.dtype, major_kind="K", swizzle_mode=swizzle_mode
                    )
                    ret[a_tensor] = a_layout_canonical.as_shared_layout().simplify()
                except ValueError:
                    continue
                else:
                    break
        if not b_tensor.has_layout():
            for swizzle_mode in [
                Tcgen05SwizzleMode.B128_SWIZZLE,
                Tcgen05SwizzleMode.B64_SWIZZLE,
                Tcgen05SwizzleMode.B32_SWIZZLE,
                Tcgen05SwizzleMode.NO_SWIZZLE,
            ]:
                try:
                    b_layout_canonical = generate_canonical_layout(
                        shape=(n, k), dtype=b_tensor.dtype, major_kind="K", swizzle_mode=swizzle_mode
                    )
                    ret[b_tensor] = b_layout_canonical.as_shared_layout().permute(dims=[1, 0]).simplify()
                except ValueError:
                    continue
                else:
                    break
        if not d_tensor.has_layout():
            d_layout = generate_wgmma_register_layout(n, d_tensor.dtype)
            ret[d_tensor] = d_layout
        return ret