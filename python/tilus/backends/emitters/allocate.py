from hidet.ir.expr import tensor_var
from hidet.ir.tools import rewrite
from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.ir import RegisterValue
from tilus.ir.inst import AllocateInst
from tilus.target import gpgpu_any


@register_inst_emitter(AllocateInst, target=gpgpu_any)
class AllocateInstEmitter(BaseInstEmitter):
    def emit(self, inst: AllocateInst):  # type: ignore
        output: RegisterValue = inst.register_output
        var = self.declare(tensor_var("regs", shape=[output.size], dtype=output.dtype))
        if inst.init is not None:
            axes, init_expr = inst.init
            with self.for_range(output.size) as i:
                global_indices = output.layout.local2global(local_index=i, worker=self.current_worker)
                self.buffer_store(
                    buf=var,
                    indices=[i],
                    value=rewrite(
                        init_expr,
                        rewrite_map={axis: global_index for axis, global_index in zip(axes, global_indices)},
                    ),
                )
        self.value2var[output] = var
