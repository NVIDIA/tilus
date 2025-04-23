from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.ir.instructions import RepeatInst, RepeatInterleaveInst, SqueezeInst, UnsqueezeInst


@register_emitter(RepeatInst)
class RepeatInstEmitter(BaseInstEmitter):
    def emit(self, inst: RepeatInst) -> None:
        src = inst.register_input
        dst = inst.register_output

        src_buf = self.tensor2var[src]
        dst_buf = self.get_or_allocate_var(dst)

        # check the layout is valid
        assert len(src.shape) == len(dst.shape)
        repeats = [a // b for a, b in zip(dst.shape, src.shape)]
        # t1 = time.time()
        # for worker in range(self.current_num_workers):
        #     for i in range(dst.local_size):
        #         global_indices = dst.layout.local2global(as_expr(i), as_expr(worker))
        #         global_indices = [a // b for a, b in zip(global_indices, repeats)]
        #         is_valid = src.layout.is_valid(global_indices, worker=as_expr(worker))
        #         if not is_valid:
        #             raise RuntimeError(
        #                 f"Invalid global index {global_indices} for worker {worker} in RepeatInst'\n"
        #                 f"src_layout: {src.layout} \n"
        #                 f"dst_layout: {dst.layout}"
        #             )
        # t2 = time.time()
        # print(f"Check layout time: {t2 - t1:.4f} seconds")

        # emit the code
        with self.for_range(dst.local_size, attr="u") as local:
            global_indices = dst.layout.local2global(local_index=local, worker=self.current_worker)
            global_indices = [a // b for a, b in zip(global_indices, repeats)]
            src_local = src.layout.global2local(global_indices, worker=self.current_worker)
            self.buffer_store(dst_buf, indices=[local], value=src_buf[src_local])


@register_emitter(RepeatInterleaveInst)
class RepeatInterleaveInstEmitter(BaseInstEmitter):
    def emit(self, inst: RepeatInterleaveInst) -> None:
        src = inst.register_input
        dst = inst.register_output

        src_buf = self.tensor2var[src]
        dst_buf = self.get_or_allocate_var(dst)

        # check the layout is valid
        assert len(src.shape) == len(dst.shape)
        # t1 = time.time()
        # for worker in range(self.current_num_workers):
        #     for i in range(dst.local_size):
        #         global_indices = dst.layout.local2global(as_expr(i), as_expr(worker))
        #         global_indices = [a % b for a, b in zip(global_indices, src.shape)]
        #         is_valid = src.layout.is_valid(global_indices, worker=as_expr(worker))
        #         if not is_valid:
        #             raise RuntimeError(
        #                 f"Invalid global index {global_indices} for worker {worker} in RepeatInst'\n"
        #                 f"src_layout: {src.layout} \n"
        #                 f"dst_layout: {dst.layout}"
        #             )
        # t2 = time.time()
        # print(f"Check layout time: {t2 - t1:.4f} seconds")

        # emit the code
        with self.for_range(dst.local_size, attr="u") as local:
            global_indices = dst.layout.local2global(local_index=local, worker=self.current_worker)
            global_indices = [a % b for a, b in zip(global_indices, src.shape)]
            src_local = src.layout.global2local(global_indices, worker=self.current_worker)
            self.buffer_store(dst_buf, indices=[local], value=src_buf[src_local])


@register_emitter(UnsqueezeInst)
@register_emitter(SqueezeInst)
class SqueezeUnsqueezeInstEmitter(BaseInstEmitter):
    def emit(self, inst: SqueezeInst) -> None:
        src = inst.register_input
        dst = inst.register_output

        src_buf = self.tensor2var[src]
        dst_buf = self.get_or_allocate_var(dst)

        # emit the code
        with self.for_range(dst.local_size, attr="u") as local:
            self.buffer_store(dst_buf, indices=[local], value=src_buf[local])
