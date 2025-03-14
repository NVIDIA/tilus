import operator

from hidet.ir.expr import Expr, Var, if_then_else, tensor_var
from hidet.ir.utils.broadcast_utils import broadcast_indices
from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.ir.inst import ElementwiseUnaryInst, ElementwiseBinaryInst, BroadcastElementwiseBinaryInst
from tilus.ir.value import RegisterValue
from tilus.target import gpgpu_any


@register_inst_emitter(ElementwiseUnaryInst, target=gpgpu_any)
class ElementwiseUnaryInstEmitter(BaseInstEmitter):
    def emit(self, inst: ElementwiseUnaryInst):
        name_mapping = {"relu": "relu", "clip": "clipped"}
        op_var_name = name_mapping[inst.op]

        x_value: RegisterValue = inst.inputs[0].as_register_value()
        y_value: RegisterValue = inst.register_output
        x_var: Var = self.value2var[x_value]
        y_var: Var = self.declare(tensor_var(op_var_name, shape=[y_value.size], dtype=y_value.dtype))
        self.value2var[y_value] = y_var

        with self.for_range(extent=y_value.size) as i:
            op_map = {
                "relu": lambda x: if_then_else(x > x_value.dtype.zero, x, x_value.dtype.zero),
                "clip": lambda x: self._clip(x, inst.attrs["min_value"], inst.attrs["max_value"]),
            }
            op = op_map[inst.op]

            self.buffer_store(buf=y_var, indices=[i], value=op(x_var[i]))

    def _clip(self, x: Expr, min_value: Expr, max_value: Expr) -> Expr:
        x = if_then_else(x < min_value, min_value, x)
        x = if_then_else(x > max_value, max_value, x)
        return x


@register_inst_emitter(ElementwiseBinaryInst, target=gpgpu_any)
class ElementwiseBinaryInstEmitter(BaseInstEmitter):
    def emit(self, inst: ElementwiseBinaryInst):
        name_mapping = {"+": "added", "-": "diff", "*": "product", "/": "quotient"}

        x_value: RegisterValue = inst.inputs[0].as_register_value()
        y_value: RegisterValue = inst.inputs[1].as_register_value()
        z_value: RegisterValue = inst.register_output
        x_var: Var = self.value2var[x_value]
        y_var: Var = self.value2var[y_value]
        z_var = self.get_or_allocate_var(z_value, name_mapping[inst.op])
        with self.for_range(extent=z_value.size) as i:
            z_indices = z_value.layout.local2global(local_index=i, worker=self.current_worker)
            x_indices = broadcast_indices(out_indices=z_indices, shape=x_value.shape, out_shape=z_value.shape)
            y_indices = broadcast_indices(out_indices=z_indices, shape=y_value.shape, out_shape=z_value.shape)
            x_local = x_value.layout.global2local(x_indices, worker=self.current_worker)
            y_local = y_value.layout.global2local(y_indices, worker=self.current_worker)

            op_map = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "%": operator.mod}
            op = op_map[inst.op]

            self.buffer_store(buf=z_var, indices=[i], value=op(x_var[x_local], y_var[y_local]))


@register_inst_emitter(BroadcastElementwiseBinaryInst, target=gpgpu_any)
class BroadcastElementwiseBinaryInstEmitter(BaseInstEmitter):
    def emit(self, inst: BroadcastElementwiseBinaryInst):
        name_mapping = {"+": "added", "-": "diff", "*": "product", "/": "quotient"}
        op_var_name = name_mapping[inst.op]

        r_value: RegisterValue = inst.inputs[0].as_register_value()
        s_expr: Expr = inst.s
        z_value: RegisterValue = inst.register_output
        r_var: Var = self.value2var[r_value]
        z_var: Var
        if z_value in self.value2var:
            z_var = self.value2var[z_value]
        else:
            z_var = self.declare(tensor_var(op_var_name, shape=[z_value.size], dtype=z_value.dtype))
            self.value2var[z_value] = z_var
        with self.for_range(extent=z_value.size) as i:
            op_map = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "%": operator.mod}

            def expr_op(x, y):
                nonlocal op_map
                if inst.tensor_left:
                    return op_map[inst.op](x, y)
                else:
                    return op_map[inst.op](y, x)

            self.buffer_store(buf=z_var, indices=[i], value=expr_op(r_var[i], s_expr))
