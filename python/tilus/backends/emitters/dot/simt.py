from typing import Tuple

from hidet.ir.expr import Expr, cast
from hidet.ir.utils.broadcast_utils import broadcast_indices
from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.ir.inst import SimtDotInst
from tilus.target import gpgpu_any


@register_inst_emitter(SimtDotInst, target=gpgpu_any)
class MmaDotInstEmitter(BaseInstEmitter):
    def emit(self, inst: SimtDotInst):  # type: ignore
        assert inst.output is inst.inputs[2]
        a_value = inst.inputs[0].as_register_value()
        b_value = inst.inputs[1].as_register_value()
        c_value = inst.inputs[2].as_register_value()
        a_buf = self.value2var[a_value]
        b_buf = self.value2var[b_value]
        c_buf = self.value2var[c_value]

        warp_id: Expr = self.current_worker // 32
        warp_spatial: Tuple[int, int, int] = inst.warp_spatial
        warp_repeat: Tuple[int, int, int] = inst.warp_repeat
        thread_spatial: Tuple[int, int] = inst.thread_spatial
        thread_repeat: Tuple[int, int] = inst.thread_repeat
        c_outer_shape = c_value.shape[:-2]

        simt_m = thread_spatial[0] * thread_repeat[0]
        simt_n = thread_spatial[1] * thread_repeat[1]
        simt_k = 1

        assert a_value.dtype == b_value.dtype
        ab_dtype = a_value.dtype
        c_dtype = c_value.dtype

        with self.for_grid(c_outer_shape) as c_outer_indices:  # type: ignore
            a_outer_indices = broadcast_indices(c_outer_indices, a_value.shape[:-2], c_outer_shape)
            b_outer_indices = broadcast_indices(c_outer_indices, b_value.shape[:-2], c_outer_shape)
            with self.for_grid(list(warp_repeat)) as repeat_indices:
                from hidet.ir.mapping import spatial_map

                spatial_indices: Tuple[Expr, Expr, Expr] = spatial_map(warp_spatial, ranks=[1, 2, 0])(warp_id)[0]

                mma_indices = [
                    (spatial_indices[0] * warp_repeat[0] + repeat_indices[0]) * simt_m,
                    (spatial_indices[1] * warp_repeat[1] + repeat_indices[1]) * simt_n,
                    (spatial_indices[2] * warp_repeat[2] + repeat_indices[2]) * simt_k,
                ]

                with self.for_grid(thread_repeat) as (i, j):  # type: ignore
                    k = 0
                    a_indices = a_outer_indices + [mma_indices[0] + i, mma_indices[2] + k]
                    b_indices = b_outer_indices + [mma_indices[2] + k, mma_indices[1] + j]
                    c_indices = c_outer_indices + [mma_indices[0] + i, mma_indices[1] + j]

                    a_local = a_value.layout.global2local(a_indices, self.current_worker)
                    b_local = b_value.layout.global2local(b_indices, self.current_worker)
                    c_local = c_value.layout.global2local(c_indices, self.current_worker)

                    aa = a_buf[a_local]
                    bb = b_buf[b_local]
                    cc = c_buf[c_local]
                    if ab_dtype != c_dtype:
                        aa = cast(aa, c_dtype)
                        bb = cast(bb, c_dtype)

                    self.buffer_store(c_buf, indices=[c_local], value=cc + aa * bb)
