from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.ir.inst import AssignInst
from tilus.target import gpgpu_any


@register_inst_emitter(AssignInst, target=gpgpu_any)
class AllocateInstEmitter(BaseInstEmitter):
    def emit(self, inst: AssignInst) -> None:  # type: ignore
        value = inst.register_output
        input_value = inst.inputs[0].as_register_tensor()
        var = self.get_or_allocate_var(tensor=value, name="regs")
        assert input_value.dtype == value.dtype
        assert input_value.layout.quick_equal(value.layout)
        with self.for_range(value.layout.local_size) as i:
            self.buffer_store(buf=var, indices=[i], value=self.tensor2var[input_value][i])

        self.tensor2var[value] = var
