from typing import Sequence
from hidet.ir.expr import Expr
from hidet.ir.type import DataType
from tilus.ir.tensor import RegisterTensor
from tilus.ir.builders import StmtBuilder


class ReigsterTensorWithMethods(RegisterTensor):
    def __init__(self, tensor: RegisterTensor, builder: StmtBuilder):
        super().__init__(tensor.dtype, tensor.shape, tensor.optional_layout)
        self.tensor: RegisterTensor = tensor
        self.builder: StmtBuilder = builder
    
    def __neg__(self) -> RegisterTensor:
        return self.builder.neg(self.tensor)
    
    def __add__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other))
        return self.builder.add(self.tensor, other)
    
    def __sub__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other))
        return self.builder.sub(self.tensor, other)
    
    def __mul__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other))
        return self.builder.mul(self.tensor, other)
    
    def __truediv__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other))
        return self.builder.div(self.tensor, other)
    
    def __ge__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other))
        return self.builder.greater_equal(self.tensor, other)
    
    def __le__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other))
        return self.builder.less_equal(self.tensor, other)

    def __gt__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other))
        return self.builder.greater_than(self.tensor, other)
    
    def __lt__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other))
        return self.builder.less_than(self.tensor, other)

    def __eq__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other))
        return self.builder.equal(self.tensor, other)
    
    def __ne__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other))
        return self.builder.not_equal(self.tensor, other)

    def __ne__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        if not isinstance(other, RegisterTensor):
            other = self.builder.allocate_register(dtype=self.tensor.dtype, shape=self.tensor.shape, f_init=lambda _: self.tensor.dtype(other))
        return self.builder.not_equal(self.tensor, other)
    
    def squeeze(self, dim: int | Sequence[int]) -> RegisterTensor:
        return self.builder.squeeze(self.tensor, dim)

    def unsqueeze(self, dim: int | Sequence[int]) -> RegisterTensor:
        return self.builder.unsqueeze(self.tensor, dim)

    def transpose(self) -> RegisterTensor:
        return self.builder.transpose(self.tensor)

    def to(self, dtype: DataType) -> RegisterTensor:
        return self.builder.cast(self.tensor, dtype=dtype)
    