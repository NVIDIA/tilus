from typing import Tuple

from hidet.ir.dtypes import uint32
from hidet.ir.expr import Expr, var, cast
from hidet.ir.primitives.cuda.mma import MmaConfig as HidetMmaConfig
from hidet.ir.utils.broadcast_utils import broadcast_indices
from tilus.extensions.hidet.ir.primitives.cuda.mma import mma_sync_v2
from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.ir.instructions import MmaDotInst, MmaConfig
from tilus.target import nvgpu_sm70


@register_inst_emitter(MmaDotInst, target=nvgpu_sm70)
class MmaDotInstEmitter(BaseInstEmitter):
    def emit(self, inst: MmaDotInst):  # type: ignore
        assert inst.output is inst.inputs[2]
        mma: MmaConfig = MmaConfig.from_name(inst.mma_inst)
        a_value = inst.inputs[0].as_register_value()
        b_value = inst.inputs[1].as_register_value()
        c_value = inst.inputs[2].as_register_value()
        a_buf = self.value2var[a_value]
        b_buf = self.value2var[b_value]
        c_buf = self.value2var[c_value]

        warp_id: Expr = self.current_worker // 32
        warp_spatial: Tuple[int, int, int] = inst.warp_spatial
        warp_repeat: Tuple[int, int, int] = inst.warp_repeat
        c_outer_shape = c_value.shape[:-2]

        with self.for_grid(c_outer_shape) as c_outer_indices:  # type: ignore
            a_outer_indices = broadcast_indices(c_outer_indices, a_value.shape[:-2], c_outer_shape)
            b_outer_indices = broadcast_indices(c_outer_indices, b_value.shape[:-2], c_outer_shape)
            with self.for_grid(list(warp_repeat)) as repeat_indices:
                from hidet.ir.mapping import spatial_map

                spatial_indices: Tuple[Expr, Expr, Expr] = spatial_map(warp_spatial, ranks=[1, 2, 0])(warp_id)[0]

                mma_indices = [
                    (spatial_indices[0] * warp_repeat[0] + repeat_indices[0]) * mma.m,
                    (spatial_indices[1] * warp_repeat[1] + repeat_indices[1]) * mma.n,
                    (spatial_indices[2] * warp_repeat[2] + repeat_indices[2]) * (mma.k * mma.vec_k),
                ]

                a_indices = a_outer_indices + [mma_indices[0], mma_indices[2]]
                b_indices = b_outer_indices + [mma_indices[2], mma_indices[1]]
                c_indices = c_outer_indices + [mma_indices[0], mma_indices[1]]

                a_regs = self.declare(
                    var("a_regs", ~uint32),
                    init=cast(~a_buf[a_value.layout.global2local(a_indices, worker=self.current_worker)], ~uint32),
                )
                b_regs = self.declare(
                    var("b_regs", ~uint32),
                    init=cast(~b_buf[b_value.layout.global2local(b_indices, worker=self.current_worker)], ~uint32),
                )
                c_regs = self.declare(
                    var("c_regs", ~uint32),
                    init=cast(~c_buf[c_value.layout.global2local(c_indices, worker=self.current_worker)], ~uint32),
                )

                with self.for_range(mma.vec_k) as vk:
                    hidet_mma: HidetMmaConfig = mma.hidet_mma_config()
                    self.append(
                        mma_sync_v2(
                            config=hidet_mma,
                            a_reg_p=[a_regs + i * mma.vec_k + vk for i in range(hidet_mma.a_regs)],
                            b_reg_p=[b_regs + i * mma.vec_k + vk for i in range(hidet_mma.b_regs)],
                            c_reg_p=[c_regs + i for i in range(hidet_mma.c_regs)],
                        )
                    )
