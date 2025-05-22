from typing import Mapping

from hidet.ir.dtypes import uint32
from hidet.ir.expr import Expr, cast, deref
from hidet.ir.node import Node
from hidet.ir.primitives.cuda.mma import ldmatrix
from hidet.ir.tools import rewrite
from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.ir.instructions import LoadMatrixInst
from tilus.ir.layout import RegisterLayout, divide
from tilus.ir.utils import vector
from tilus.target import nvgpu_sm75
from tilus.utils import gcd


@register_emitter(LoadMatrixInst, target=nvgpu_sm75)
class AllocateInstEmitter(BaseInstEmitter):
    def emit(self, inst: LoadMatrixInst) -> None:
        tensor = inst.register_output
        layout = tensor.layout
        ldmatrix_layout = inst.config.ldmatrix_layout

        lhs_layout: RegisterLayout = divide(layout, ldmatrix_layout)

        regs_buf = self.get_or_allocate_var(tensor)

        vector_size: int = gcd(lhs_layout.local_size, 4)
        num_vectors: int = lhs_layout.local_size // vector_size

        with self.for_range(num_vectors) as vec_i:
            # load vector_size times of 8x16 bytes of data for each iteration,

            # get the registers
            regs: list[Expr] = []
            for i in range(vector_size):
                regs.append(deref(cast(~regs_buf[(vec_i * vector_size + i) * ldmatrix_layout.local_size], ~uint32)))

            # get the address of each row
            lane_id = self.current_worker % 32
            warp_id = self.current_worker // 32
            lhs_indices = vector(
                lhs_layout.get_global(local_index=vec_i * vector_size + lane_id // 8, spatial_index=warp_id)
            )
            rhs_indices = vector([lane_id % 8, 0])
            rhs_shape = vector(ldmatrix_layout.shape)
            shared_indices = list(lhs_indices * rhs_shape + rhs_indices)

            rewrite_map: Mapping[Node, Node] = {axis: index for axis, index in zip(inst.axes, shared_indices)}
            offset = rewrite(inst.offset, rewrite_map=rewrite_map)
            smem_addr = inst.ptr + offset

            self.append(ldmatrix(regs=regs, smem_addr=smem_addr, trans=inst.config.trans))
