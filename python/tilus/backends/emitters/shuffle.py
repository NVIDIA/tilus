from hidet.ir.dtypes import boolean, uint8, uint32
from hidet.ir.expr import Expr, Var, bitwise_and, cast, left_shift, logical_and, tensor_pointer_var
from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.extensions.hidet.ir.primitives.cuda.ldst import load, store
from tilus.ir.inst import ShuffleBaseInst, ShuffleDownInst, ShuffleUpInst
from tilus.target import nvgpu_any
from tilus.utils import gcd


@register_inst_emitter(ShuffleUpInst, target=nvgpu_any)
@register_inst_emitter(ShuffleDownInst, target=nvgpu_any)
class ShuffleBaseInstEmitter(BaseInstEmitter):
    def emit(self, inst: ShuffleBaseInst) -> None:
        dtype = inst.register_output.dtype
        layout = inst.register_output.layout
        thread_nbytes: int = dtype.nbytes * layout.local_size
        num_groups = max([i // inst.width for i in range(self.num_warps) if inst.mask & (1 << i)]) + 1
        warp_nbytes: int = thread_nbytes * 32
        assert self.codegen.smem_workspace is not None
        smem_buf: Var = self.declare(
            v=tensor_pointer_var("shfl_smem", shape=[num_groups, inst.width - inst.delta, warp_nbytes], dtype=uint8),
            init=cast(self.value2var[self.codegen.smem_workspace], ~uint8),
        )
        warp_id: Expr = self.current_worker // 32
        warp_lane_id = self.current_worker % 32
        group_id = warp_id // inst.width
        shfl_lane_id = warp_id % inst.width

        cond_in_mask = cast(bitwise_and(inst.mask, left_shift(1, warp_id)), boolean)
        if isinstance(inst, ShuffleDownInst):
            cond_is_sender = shfl_lane_id >= inst.delta
            cond_is_receiver = shfl_lane_id < inst.width - inst.delta
            sender_shfl_lane = shfl_lane_id - inst.delta
            receiver_shfl_lane = shfl_lane_id
        else:
            cond_is_sender = shfl_lane_id < inst.width - inst.delta
            cond_is_receiver = shfl_lane_id >= inst.delta
            sender_shfl_lane = shfl_lane_id
            receiver_shfl_lane = shfl_lane_id - inst.delta
        # store the data from the register to shared memory
        src_var: Var = self.value2var[inst.inputs[0]]
        with self.if_then(logical_and(cond_in_mask, cond_is_sender)):
            if thread_nbytes % 4 == 0:
                vec: int = gcd(thread_nbytes // 4, 4)
                vec_nbytes: int = vec * 4

                with self.for_range(thread_nbytes // vec_nbytes) as i:
                    self.append(
                        store(
                            dtype=uint32,
                            addr=~smem_buf[
                                group_id, sender_shfl_lane, i * (32 * vec_nbytes) + warp_lane_id * vec_nbytes
                            ],
                            src_addrs=[cast(src_var, ~uint8) + j * 4 for j in range(vec)],
                        )
                    )
            else:
                raise NotImplementedError()
        self.sync()

        # load the data from shared memory to register
        dst_var: Var = self.get_or_allocate_var(inst.register_output, "shuffled")
        with self.if_then(logical_and(cond_in_mask, cond_is_receiver)):
            if thread_nbytes % 4 == 0:
                vec = gcd(thread_nbytes // 4, 4)
                vec_nbytes = vec * 4

                with self.for_range(thread_nbytes // vec_nbytes) as i:
                    self.append(
                        load(
                            dtype=uint32,
                            addr=~smem_buf[
                                group_id, receiver_shfl_lane, i * (32 * vec_nbytes) + warp_lane_id * vec_nbytes
                            ],
                            dst_addrs=[cast(dst_var, ~uint8) + j * 4 for j in range(vec)],
                        )
                    )
            else:
                raise NotImplementedError()
        self.sync()
