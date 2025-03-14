from hidet.ir.expr import tensor_var
from hidet.ir.tools import rewrite
from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.ir.inst import AllocateInst
from tilus.target import gpgpu_any


@register_inst_emitter(AllocateInst, target=gpgpu_any)
class AllocateInstEmitter(BaseInstEmitter):
    def emit(self, inst: AllocateInst):  # type: ignore
        value = inst.register_output
        var = self.declare(tensor_var("regs", shape=[value.size], dtype=value.dtype))
        if inst.init is not None or inst.axes is not None:
            assert inst.axes is not None and inst.init is not None
            with self.for_range(value.size) as i:
                global_indices = value.layout.local2global(local_index=i, worker=self.current_worker)
                self.buffer_store(
                    buf=var,
                    indices=[i],
                    value=rewrite(
                        inst.init,
                        rewrite_map={axis: global_index for axis, global_index in zip(inst.axes, global_indices)},
                    ),
                )
        self.value2var[value] = var
