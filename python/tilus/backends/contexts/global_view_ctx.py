from dataclasses import dataclass
from hidet.ir.type import DataType
from hidet.ir.expr import Var, Expr

from tilus.ir import GlobalLayout
from tilus.ir.tensor import GlobalTensor
from tilus.backends.codegen import BaseEmitContext, FunctionCodegen


@dataclass
class GlobalTensorView:
    ptr: Var
    dtype: DataType
    layout: GlobalLayout


class GlobalTensorViewContext(BaseEmitContext):
    """ Context used to track the global tensor views that takes kernel parameters as ptr. """

    def __init__(self, codegen: FunctionCodegen):
        super().__init__(codegen)
        self.tensor2view: dict[GlobalTensor, GlobalTensorView] = {}

    def add_tensor_view(self, tensor: GlobalTensor, ptr: Var, layout: GlobalLayout):
        assert tensor not in self.tensor2view
        self.tensor2view[tensor] = GlobalTensorView(ptr, tensor.dtype, layout)
