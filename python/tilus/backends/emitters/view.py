from hidet.ir.expr import Var, tensor_pointer_var, cast
from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.ir.instructions import ViewInst
from tilus.target import gpgpu_any


@register_inst_emitter(ViewInst, target=gpgpu_any)
class ViewInstEmitter(BaseInstEmitter):
    def emit(self, inst: ViewInst):
        out_value = inst.register_output
        in_var = self.value2var[inst.inputs[0]]
        out_var: Var = self.declare(
            v=tensor_pointer_var("viewed", shape=[out_value.layout.local_size], dtype=out_value.dtype),
            init=cast(~in_var[inst.local_offset], ~out_value.dtype),
        )
        self.value2var[out_value] = out_var
