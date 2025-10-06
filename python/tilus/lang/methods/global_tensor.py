from tilus.ir.builders import StmtBuilder
from tilus.ir.tensor import GlobalTensor


class GlobalTensorWithMethods(GlobalTensor):
    def __init__(self, tensor: GlobalTensor, builder: StmtBuilder):
        super().__init__(tensor.dtype, tensor.layout)
        self.tensor = tensor
        self.builder = builder
