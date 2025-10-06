from tilus.ir.builders import StmtBuilder
from tilus.ir.tensor import SharedTensor
from tilus.lang.methods.exception import TensorMethodError


class SharedTensorWithMethods(SharedTensor):
    def __init__(self, tensor: SharedTensor, builder: StmtBuilder):
        super().__init__(tensor.dtype, tensor.shape, tensor.optional_layout)
        self.tensor: SharedTensor = tensor
        self.builder: StmtBuilder = builder

    def permute(self, dims: tuple[int, ...]) -> SharedTensor:
        if set(dims) != set(range(len(self.tensor.shape))):
            raise TensorMethodError(f"Dims must be a permutation of {range(len(self.tensor.shape))}, got {dims}")
        return self.builder.permute_shared(self.tensor, dims)

    def transpose(self) -> SharedTensor:
        return self.builder.permute_shared(self.tensor, dims=[1, 0])
