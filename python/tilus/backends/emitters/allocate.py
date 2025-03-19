from hidet.ir.expr import tensor_var
from hidet.ir.tools import rewrite
from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.ir import RegisterTensor
from tilus.ir.inst import AllocateRegisterInst
from tilus.target import gpgpu_any


@register_inst_emitter(AllocateRegisterInst, target=gpgpu_any)
class AllocateInstEmitter(BaseInstEmitter):
    def emit(self, inst: AllocateRegisterInst) -> None:  # type: ignore
        output: RegisterTensor = inst.register_output
        var = self.declare(tensor_var("regs", shape=[output.local_size], dtype=output.dtype))
        if inst.init is not None:
            axes, init_expr = inst.init
            with self.for_range(output.local_size) as i:
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
