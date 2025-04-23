from hidet.ir.expr import Var
from hidet.ir.utils.broadcast_utils import broadcast_indices
from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.ir.instructions import ElementwiseBinaryBaseInst, ElementwiseUnaryBaseInst
from tilus.ir.tensor import RegisterTensor


@register_emitter(ElementwiseUnaryBaseInst)
class ElementwiseUnaryInstEmitter(BaseInstEmitter):
    def emit(self, inst: ElementwiseUnaryBaseInst) -> None:
        x_tensor: RegisterTensor = inst.inputs[0].as_register_tensor()
        y_tensor: RegisterTensor = inst.register_output
        x_buf = self.tensor2var[x_tensor]
        y_buf = self.get_or_allocate_var(y_tensor)

        with self.for_range(extent=y_tensor.local_size) as i:
            self.buffer_store(buf=y_buf, indices=[i], value=inst.f_compute(x_buf[i]))


@register_emitter(ElementwiseBinaryBaseInst)
class ElementwiseBinaryInstEmitter(BaseInstEmitter):
    def emit(self, inst: ElementwiseBinaryBaseInst) -> None:
        x_tensor: RegisterTensor = inst.inputs[0].as_register_tensor()
        y_tensor: RegisterTensor = inst.inputs[1].as_register_tensor()
        z_tensor: RegisterTensor = inst.register_output
        x_buf: Var = self.tensor2var[x_tensor]
        y_buf: Var = self.tensor2var[y_tensor]
        z_buf = self.get_or_allocate_var(z_tensor)
        with self.for_range(extent=z_tensor.local_size) as i:
            z_indices = z_tensor.layout.local2global(local_index=i, worker=self.current_worker)
            x_indices = broadcast_indices(out_indices=z_indices, shape=x_tensor.shape, out_shape=z_tensor.shape)
            y_indices = broadcast_indices(out_indices=z_indices, shape=y_tensor.shape, out_shape=z_tensor.shape)
            x_local = x_tensor.layout.global2local(x_indices, worker=self.current_worker)
            y_local = y_tensor.layout.global2local(y_indices, worker=self.current_worker)

            self.buffer_store(buf=z_buf, indices=[i], value=inst.f_compute(x_buf[x_local], y_buf[y_local]))
