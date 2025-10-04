from tilus.ir.tensor import SharedTensor
from tilus.ir.builders import StmtBuilder

class SharedTensorWithMethods(SharedTensor):
    def __init__(self, tensor: SharedTensor, builder: StmtBuilder):
        super().__init__(tensor.dtype, tensor.shape, tensor.optional_layout)
        self.tensor: SharedTensor = tensor
        self.builder: StmtBuilder = builder
    
    def permute(self, dims: tuple[int, ...]) -> SharedTensor:
        if set(dims) != set(range(len(self.tensor.shape))):
            raise ValueError(f"Dims must be a permutation of {range(len(self.tensor.shape))}, got {dims}")
        return self.builder.permute_shared(self.tensor, dims)
