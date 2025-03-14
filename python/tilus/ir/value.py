from __future__ import annotations
from typing import Union, List

from enum import Enum

from hidet.ir.type import DataType, PointerType
from tilus.ir.layout import Layout
from tilus.ir.shared_layout import SharedLayout
from tilus.utils import nbytes_from_nbits


class Scope(Enum):
    REGISTER = 0
    SHARED = 1


class Value:
    def __init__(self, dtype: DataType, shape: List[int]):
        self.dtype: DataType = dtype
        self.shape: List[int] = shape

    def as_register_value(self) -> RegisterValue:
        assert isinstance(self, RegisterValue)
        return self

    def as_shared_value(self) -> SharedValue:
        assert isinstance(self, SharedValue)
        return self


class RegisterValue(Value):
    def __init__(self, dtype: DataType, layout: Layout):
        super().__init__(dtype, layout.shape)
        self.size: int = layout.local_size
        self.layout: Layout = layout


class SharedValue(Value):
    def __init__(self, dtype: DataType, layout: SharedLayout):
        super().__init__(dtype, layout.shape)
        self.size: int = layout.size
        self.layout: SharedLayout = layout

    def nbytes(self):
        return nbytes_from_nbits(self.size * self.dtype.nbits)


class ScalarValue(Value):
    def __init__(self, data_type: Union[DataType, PointerType]):
        super().__init__(data_type, [])  # type: ignore
        self.data_type: Union[DataType, PointerType] = data_type
        raise ValueError("deprecated")
