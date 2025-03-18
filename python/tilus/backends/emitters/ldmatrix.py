from typing import List, Optional, Tuple, Sequence

from hidet.ir.dtypes import uint32, int32
from hidet.ir.expr import Expr, tensor_var, tensor_pointer_var, cast
from hidet.ir.primitives.cuda.mma import ldmatrix
from hidet.ir.type import DataType

from tilus.backends.codegen import BaseInstEmitter, InvalidInstruction, register_inst_emitter
from tilus.extensions.hidet.ir.utils.index_transform import index_add, index_multiply
from tilus.ir.inst import LoadMatrixInst
from tilus.ir.layout import Layout, repeat, divide, get_composition_chain, compose_chain, compose, identity, simplify
from tilus.ir.value import SharedValue, RegisterValue
from tilus.target import nvgpu_sm75
from tilus.utils import prod


@register_inst_emitter(LoadMatrixInst, target=nvgpu_sm75)
class LoadMatrixInstEmitter(BaseInstEmitter):
    @staticmethod
    def resolve(inst: LoadMatrixInst) -> Tuple[bool, Layout, Layout, Layout]:
        dst: RegisterValue = inst.register_output
        layout: Layout = dst.layout
        dtype: DataType = dst.dtype

        # check whether the layout can be loaded via ldmatrix
        for nbytes, trans, ldmatrix_layout in LoadMatrixInst.LDMATRIX_CONFIGS:
            if nbytes != dtype.nbytes:
                continue
            outer: Optional[Layout] = divide(layout, ldmatrix_layout)
            if outer is None:
                continue
            decomposed_outer: List[Layout] = get_composition_chain(outer, fine_grained=True)
            middle = repeat(1, 1)
            while len(decomposed_outer) > 0:
                tail = decomposed_outer[-1]
                if tail.num_workers == 1 and prod(tail.shape) * prod(middle.shape) in [1, 2, 4]:
                    middle = compose(tail, middle)
                    decomposed_outer.pop()
                else:
                    break
            if len(decomposed_outer) == 0:
                outer = identity(len(outer.shape))
            else:
                outer = simplify(compose_chain(decomposed_outer))

            return trans, ldmatrix_layout, middle, outer

        raise InvalidInstruction(inst)

    def emit(self, inst: LoadMatrixInst) -> None:
        trans, atom, middle, outer = LoadMatrixInstEmitter.resolve(inst)
        warp_id = self.current_worker // 32

        dst: RegisterValue = inst.register_output
        src: SharedValue = inst.inputs[0].as_shared_value()
        vec_size = 4 // dst.dtype.nbytes
        var = self.declare(v=tensor_var("ldm", shape=[dst.size], dtype=dst.dtype))
        regs_view = self.declare(
            v=tensor_pointer_var("regs", shape=[dst.size // vec_size], dtype=uint32), init=cast(var, ~uint32)
        )
        # src_buf = self.value2var[src]
        # src_shared_addr = self.declare(Var('smem_addr', int32), init=cvta_generic_to_shared(src_buf))
        src_shared_addr = self.shared_value_shared_space_addr[src]
        offsets: List[Expr] = inst.offsets

        with self.for_range(outer.local_size, attr="u+") as outer_local:
            # prepare regs
            regs: List[Expr] = []
            for middle_local in range(middle.local_size):
                outer_stride = middle.local_size * atom.local_size
                middle_stride = atom.local_size
                regs.append(regs_view[(outer_local * outer_stride + middle_local * middle_stride) // vec_size])

            # prepare shared memory addr
            lane_id = self.current_worker % 32
            outer_indices: List[Expr] = outer.local2global(outer_local, warp_id)
            outer_strides: List[Expr] = index_multiply(middle.shape, atom.shape)
            middle_indices = middle.local2global(local_index=lane_id // 8, worker=int32.zero)
            middle_strides: Sequence[int] = atom.shape
            atom_indices = [lane_id % 8, 0]
            smem_indices = index_add(
                offsets,
                index_add(
                    index_add(
                        index_multiply(outer_indices, outer_strides), index_multiply(middle_indices, middle_strides)
                    ),
                    atom_indices,
                ),
            )
            self.append(
                ldmatrix(
                    regs=regs,
                    smem_addr=src_shared_addr + src.layout(*smem_indices) * src.dtype.nbytes,
                    shared_space_addr=True,
                    trans=trans,
                )
            )

        self.value2var[dst] = var
