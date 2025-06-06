from __future__ import annotations

import functools

from hidet import boolean
from hidet.ir import DataType
from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr, Var, cast, if_then_else, logical_and
from hidet.ir.primitives.cuda.shfl import shfl_down_sync, shfl_up_sync
from hidet.ir.type import tensor_pointer_type
from hidet.utils.py import is_power_of_two

from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.extensions.hidet.ir.utils.index_transform import index_deserialize, index_serialize
from tilus.ir.instructions.generic import ReduceInst
from tilus.ir.layout import RegisterLayout, SharedLayout, shared_repeat
from tilus.ir.tensor import RegisterTensor
from tilus.target import nvgpu_any
from tilus.utils import prod


@register_emitter(ReduceInst, target=nvgpu_any)
class ReduceInstEmitter(BaseInstEmitter):
    def scalar_init_value(self, op: str, dtype: DataType) -> Expr:
        if op == "sum":
            return dtype.zero
        elif op == "max":
            return dtype.min_value
        elif op == "min":
            return dtype.max_value
        else:
            raise NotImplementedError()

    def scalar_reduce(self, lhs: Expr, rhs: Expr, op: str) -> Expr:
        from hidet.ir.primitives import max, min

        if op == "sum":
            return lhs + rhs
        elif op == "max":
            return max(lhs, rhs)
        elif op == "min":
            return min(lhs, rhs)
        else:
            raise NotImplementedError()

    def intra_thread_reduce(self, inst: ReduceInst) -> None:
        src: RegisterTensor = inst.register_input
        dst: RegisterTensor = inst.register_output
        dim: int = inst.dim
        layout: RegisterLayout = src.layout
        src_buf: Var = self.tensor2var[src]
        dst_buf: Var = self.get_or_allocate_var(dst)

        with self.for_range(dst.layout.local_size, attr="u") as dst_local:
            self.buffer_store(dst_buf, indices=[dst_local], value=self.scalar_init_value(inst.op, dst.dtype))

        reduced_modes = layout.grouped_modes[dim]
        reduced_local_dims = [dim for dim, mode in enumerate(layout.local_modes) if mode in reduced_modes]

        src_local_shape = src.layout.local_shape
        dst_local_shape = [size for dim, size in enumerate(src_local_shape) if dim not in reduced_local_dims]
        with self.for_range(src.layout.local_size, attr="u") as src_local:
            src_local_indices = index_deserialize(src_local, shape=src_local_shape)
            dst_local_indices = [index for dim, index in enumerate(src_local_indices) if dim not in reduced_local_dims]
            dst_local = index_serialize(dst_local_indices, shape=dst_local_shape)
            self.buffer_store(
                dst_buf, indices=[dst_local], value=self.scalar_reduce(dst_buf[dst_local], src_buf[src_local], inst.op)
            )

    @staticmethod
    def check_whether_spatial_bit_reduced(spatial_bit: int, layout: RegisterLayout, dim: int) -> bool:
        """
        Check whether the spatial bit is reduced given the layout and reduction dimension.
        """
        # find first mode that its accumulation is larger or equal to 2^lane_bit, which is the mode that contains the
        # lane_bit
        accumulation = 1
        reduced_modes = layout.grouped_modes[dim]
        for mode in reversed(layout.spatial_modes):
            mode_size = layout.mode_shape[mode] if mode >= 0 else -mode
            accumulation *= mode_size
            if accumulation > (1 << spatial_bit):
                # we found the mode that contains the spatial_bit, and need to check whether it is reduced or not
                return mode in reduced_modes
        assert False, "should never reach here."

    @staticmethod
    def check_whether_spatial_bit_replicated(spatial_bit: int, layout: RegisterLayout) -> bool:
        """
        Check whether the spatial bit is replicated given the layout and reduction dimension.
        """
        # find first mode that its accumulation is larger or equal to 2^lane_bit, which is the mode that contains the
        # lane_bit
        accumulation = 1
        for mode in reversed(layout.spatial_modes):
            mode_size = layout.mode_shape[mode] if mode >= 0 else -mode
            accumulation *= mode_size
            if accumulation > (1 << spatial_bit):
                # we found the mode that contains the spatial_bit, and need to check whether it is replicated or not
                return mode < 0
        assert False, "should never reach here."

    def intra_warp_reduce(self, inst: ReduceInst) -> None:
        src: RegisterTensor = inst.register_input
        dst: RegisterTensor = inst.register_output
        dim: int = inst.dim
        layout: RegisterLayout = src.layout
        dst_buf = self.tensor2var[dst]
        warp_nbits = 5  # warp_nbits = log2(warp_size) = 5

        with self.for_range(dst.layout.local_size, attr="u") as dst_local:
            for lane_bit in range(warp_nbits):
                if not self.check_whether_spatial_bit_reduced(lane_bit, layout, dim):
                    continue
                self.buffer_store(
                    buf=dst_buf,
                    indices=[dst_local],
                    value=self.scalar_reduce(
                        lhs=dst_buf[dst_local],
                        rhs=shfl_down_sync(
                            mask=0xFFFFFFFF, var=dst_buf[dst_local], delta=1 << lane_bit, width=1 << (lane_bit + 1)
                        ),
                        op=inst.op,
                    ),
                )

    def intra_warp_broadcast(self, inst: ReduceInst) -> None:
        src: RegisterTensor = inst.register_input
        dst: RegisterTensor = inst.register_output
        dim: int = inst.dim
        layout: RegisterLayout = src.layout
        dst_buf = self.tensor2var[dst]
        warp_nbits = 5  # warp_nbits = log2(warp_size) = 5

        with self.for_range(dst.layout.local_size, attr="u") as dst_local:
            for lane_bit in reversed(range(warp_nbits)):
                if not self.check_whether_spatial_bit_reduced(lane_bit, layout, dim):
                    continue
                self.buffer_store(
                    buf=dst_buf,
                    indices=[dst_local],
                    value=shfl_up_sync(
                        mask=0xFFFFFFFF, var=dst_buf[dst_local], delta=1 << lane_bit, width=1 << (lane_bit + 1)
                    ),
                )

    @staticmethod
    def requires_inter_warp_reduction(inst: ReduceInst) -> bool:
        layout = inst.register_input.layout
        num_warps = layout.spatial_size // 32
        reduce_modes = layout.grouped_modes[inst.dim]
        accumulation = 1
        for mode in layout.spatial_modes:
            if mode >= 0 and mode in reduce_modes and accumulation < num_warps:
                # this mode is reduced, and (part of) it is in the warp dimension, we need inter-warp reduction
                return True
            mode_size = layout.mode_shape[mode] if mode >= 0 else -mode
            accumulation *= mode_size
        return False

    @staticmethod
    @functools.lru_cache(maxsize=4)
    def analyze_modes(inst):
        layout = inst.register_input.layout
        dim = inst.dim
        reduced_modes = layout.grouped_modes[dim]
        spatial_size = layout.spatial_size
        num_warps = spatial_size // 32
        warp_mode_shape = []
        warp_mode_kinds = []  # "replicated", "reduced", "spatial"
        lane_mode_shape = []
        lane_mode_kinds = []  # "replicated", "reduced", "spatial"
        existing_size = 1
        for mode in layout.spatial_modes:
            mode_size = layout.mode_shape[mode] if mode >= 0 else -mode
            if mode < 0:
                kind = "replicated"
            elif mode in reduced_modes:
                kind = "reduced"
            else:
                kind = "spatial"

            if existing_size < num_warps < existing_size * mode_size:
                # this mode cross the boundary of warp & lane, we split it into two parts
                warp_mode_size = num_warps // existing_size
                lane_mode_size = mode_size // warp_mode_size
                assert warp_mode_size * lane_mode_size == mode_size and warp_mode_size > 1 and lane_mode_size > 1
                warp_mode_shape.append(warp_mode_size)
                warp_mode_kinds.append(kind)
                lane_mode_shape.append(lane_mode_size)
                lane_mode_kinds.append(kind)
            elif existing_size * mode_size <= num_warps:
                # this mode is a warp mode
                warp_mode_shape.append(mode_size)
                warp_mode_kinds.append(kind)
            else:
                # this mode is a lane mode
                lane_mode_shape.append(mode_size)
                lane_mode_kinds.append(kind)
            existing_size *= mode_size
        return warp_mode_shape, warp_mode_kinds, lane_mode_shape, lane_mode_kinds

    def determine_shared_layout(self, inst: ReduceInst) -> SharedLayout:
        warp_mode_shape, warp_mode_kinds, lane_mode_shape, lane_mode_kinds = self.analyze_modes(inst)
        warp_part_size = prod(
            [mode_size for mode_size, mode_kind in zip(warp_mode_shape, warp_mode_kinds) if mode_kind == "spatial"]
        )
        lane_part_size = prod(
            [mode_size for mode_size, mode_kind in zip(lane_mode_shape, lane_mode_kinds) if mode_kind == "spatial"]
        )
        local_part_size = inst.register_output.layout.local_size
        return shared_repeat(warp_part_size, lane_part_size, local_part_size)

    def get_smem_idx(self, inst, warp_id, lane_id, local_id, shared_layout):
        warp_mode_shape, warp_mode_kinds, lane_mode_shape, lane_mode_kinds = self.analyze_modes(inst)

        # extract the warp, lane index after filtering the replicated and reduced parts
        warp_indices = index_deserialize(warp_id, shape=warp_mode_shape)
        warp_id_reduced = []
        warp_shape_reduced = []
        for mode_kind, mode_index, mode_size in zip(warp_mode_kinds, warp_indices, warp_mode_shape):
            if mode_kind == "spatial":
                warp_shape_reduced.append(mode_size)
                warp_id_reduced.append(mode_index)

        lane_indices = index_deserialize(lane_id, lane_mode_shape)
        lane_id_reduced = []
        lane_shape_reduced = []
        for mode_kind, mode_index, mode_size in zip(lane_mode_kinds, lane_indices, lane_mode_shape):
            if mode_kind == "spatial":
                lane_shape_reduced.append(mode_size)
                lane_id_reduced.append(mode_index)

        warp_idx = index_serialize(warp_id_reduced, shape=warp_shape_reduced)
        lane_idx = index_serialize(lane_id_reduced, shape=lane_shape_reduced)
        smem_idx = shared_layout(warp_idx, lane_idx, local_id)
        return smem_idx

    def inter_warp_reduce(self, inst: ReduceInst) -> None:
        """
        Perform the inter-warp reduction using shared memory.

        1. determine the modes for the warps, and whether each mode is: 1) replicated, 2) reduced, or 3) spatial
        2. enumerate the indices for reduced modes, reduced_indices
          2.1 reduce the data (within-warp, local) in registers with the data in shared memory
          2.2 or, store the data to shared memory (if all reduced_indices are 0)
        3. for each warp and each lane, load the data from shared memory
        """
        dst: RegisterTensor = inst.register_output
        dst_buf = self.get_or_allocate_var(dst)
        lane_id = self.declare_var("lane_id", int32, self.current_worker % 32)
        warp_id = self.declare_var("warp_id", int32, self.current_worker // 32)

        # 1. determine the modes for the warps, and whether each mode is: 1) replicated, 2) reduced, or 3) spatial
        warp_mode_shape, warp_mode_kinds, lane_mode_shape, lane_mode_kinds = self.analyze_modes(inst)

        # 2. enumerate the indices for reduced modes
        shared_layout = self.determine_shared_layout(inst)  # [warp, lane, local]
        smem_buf = self.declare_var(
            "smem_buf",
            tensor_pointer_type(dtype=dst.dtype, shape=[shared_layout.size]),
            init=cast(self.tensor2var[self.codegen.smem_workspace], ~dst.dtype),
        )

        reduced_mode_shape = [
            mode_size if kind == "reduced" else 1 for kind, mode_size in zip(warp_mode_kinds, warp_mode_shape)
        ]
        with self.for_range(prod(reduced_mode_shape), attr="u") as red_i:
            warp_indices = index_deserialize(warp_id, shape=warp_mode_shape)

            no_replicated = boolean.true
            for i, mode_kind in enumerate(warp_mode_kinds):
                if mode_kind == "replicated":
                    # if any mode is replicated, we cannot reduce it, so we skip this warp
                    no_replicated = logical_and(no_replicated, warp_indices[i] == 0)

            red_i_actual = index_serialize(
                [idx if mode == "reduced" else 0 for idx, mode in zip(warp_indices, warp_mode_kinds)],
                shape=reduced_mode_shape,
            )

            # get the lane_mask that requires to be 0 (reduced or replicated)
            lane_mask = 0
            acc_size = 1
            for lane_mode_kind, lane_mode_size in reversed(list(zip(lane_mode_kinds, lane_mode_shape))):
                if lane_mode_kind in ("reduced", "replicated"):
                    assert is_power_of_two(lane_mode_size)
                    lane_mask |= (lane_mode_size - 1) * acc_size
                acc_size *= lane_mode_size

            with self.if_then(logical_and(no_replicated, red_i == red_i_actual, (lane_id & lane_mask) == 0)):
                # 2. enumerate the indices for reduced modes
                with self.for_range(dst.layout.local_size, attr="u") as dst_local:
                    smem_idx = self.declare_var(
                        "smem_idx", tp=int32, init=self.get_smem_idx(inst, warp_id, lane_id, dst_local, shared_layout)
                    )
                    value = if_then_else(
                        red_i == 0,
                        then_expr=dst_buf[dst_local],  # 2.1
                        else_expr=self.scalar_reduce(  # 2.2
                            lhs=dst_buf[dst_local], rhs=smem_buf[smem_idx], op=inst.op
                        ),
                    )
                    self.buffer_store(smem_buf, indices=[smem_idx], value=value)
            self.sync()

        # 3. for each warp and each lane, load the data from shared memory
        with self.for_range(dst.layout.local_size, attr="u") as dst_local:
            smem_idx = self.declare_var(
                "smem_idx", tp=int32, init=self.get_smem_idx(inst, warp_id, lane_id, dst_local, shared_layout)
            )
            # load the data from shared memory
            value = smem_buf[smem_idx]
            # store the data to the destination buffer
            self.buffer_store(dst_buf, indices=[dst_local], value=value)
        self.sync()

    def efficient_reduce(self, inst: ReduceInst) -> None:
        # reduce within thread
        self.intra_thread_reduce(inst)

        # reduce within warp
        self.intra_warp_reduce(inst)

        if self.requires_inter_warp_reduction(inst):
            # reduce between warps
            self.inter_warp_reduce(inst)
        else:
            # broadcast within warp, we only need to do broadcast when we did not do inter-warp reduction
            self.intra_warp_broadcast(inst)

    def request_shared_workspace(self, inst: ReduceInst) -> int:
        if self.requires_inter_warp_reduction(inst):
            # we need shared memory for inter-warp reduction
            shared_layout = self.determine_shared_layout(inst)
            return shared_layout.size * inst.register_output.dtype.nbytes
        else:
            return 0

    def emit(self, inst: ReduceInst) -> None:
        self.efficient_reduce(inst)
