from typing import List, Callable
from hidet.ir.type import DataType
from hidet.ir.dtypes import int32
from hidet.ir.expr import Var, Expr, var
from tilus.ir.layout import Layout
from tilus.utils import idiv


class WeightTransform:
    def __str__(self):
        from tilus.ir.printer import VirtualMachinePrinter

        printer = VirtualMachinePrinter()
        return str(printer.visit(self))


class IndexSymbolicMapping:
    def __init__(self, axis: Var, index: Expr):
        self.axis: Var = axis
        self.index: Expr = index

    def __call__(self, axis_value: Expr) -> Expr:
        from hidet.ir.tools import rewrite

        return rewrite(self.index, rewrite_map={self.axis: axis_value})

    @staticmethod
    def create(f_indices: Callable[[Var], Expr]):
        axis = var("i", int32)
        index = f_indices(axis)
        return IndexSymbolicMapping(axis, index)


class ValueSymbolicMapping:
    def __init__(self, x: Var, value: Expr):
        self.x: Var = x
        self.value: Expr = value

    def __call__(self, x_value):
        from hidet.ir.tools import rewrite

        return rewrite(self.value, rewrite_map={self.x: x_value})

    @staticmethod
    def create(x_dtype: DataType, f_value: Callable[[Var], Expr]):
        x = var("x", x_dtype)
        value = f_value(x)
        return ValueSymbolicMapping(x, value)


class WeightLayoutTransformGeneric(WeightTransform):
    def __init__(
        self, dtype: DataType, size: int, mapping: IndexSymbolicMapping, reverse_mapping: IndexSymbolicMapping
    ):
        self.dtype: DataType = dtype
        self.size: int = size
        self.mapping: IndexSymbolicMapping = mapping
        self.reverse_mapping: IndexSymbolicMapping = reverse_mapping

    @staticmethod
    def create(dtype: DataType, size: int, f_apply: Callable[[Var], Expr], f_reverse: Callable[[Var], Expr]):
        return WeightLayoutTransformGeneric(
            dtype, size, IndexSymbolicMapping.create(f_apply), IndexSymbolicMapping.create(f_reverse)
        )


class WeightLayoutTransform(WeightTransform):
    def __init__(
        self,
        dtype: DataType,
        shape: List[int],
        strides: List[int],
        original_layout: Layout,
        transformed_dtype: DataType,
        transformed_layout: Layout,
    ):
        super().__init__()
        self.dtype: DataType = dtype
        self.shape: List[int] = shape
        self.strides: List[int] = strides
        self.original_layout: Layout = original_layout
        self.transformed_dtype: DataType = transformed_dtype
        self.transformed_layout: Layout = transformed_layout

        self.tile_shape: List[int] = list(original_layout.shape)
        self.num_tiles: List[int] = [idiv(a, b) for a, b in zip(self.shape, self.original_layout.shape)]


class WeightValueTransform(WeightTransform):
    def __init__(
        self, dtype: DataType, shape: List[int], mapping: ValueSymbolicMapping, reverse_mapping: ValueSymbolicMapping
    ):
        self.dtype: DataType = dtype
        self.shape: List[int] = shape
        self.mapping: ValueSymbolicMapping = mapping
        self.reverse_mapping: ValueSymbolicMapping = reverse_mapping

    @staticmethod
    def create(dtype: DataType, shape: List[int], f_apply: Callable[[Var], Expr], f_reverse: Callable[[Var], Expr]):
        return WeightValueTransform(
            dtype, shape, ValueSymbolicMapping.create(dtype, f_apply), ValueSymbolicMapping.create(dtype, f_reverse)
        )
