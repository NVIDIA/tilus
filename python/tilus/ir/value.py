from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from hidet.ir.type import DataType
from tilus.ir.layout import Layout, SharedLayout
from tilus.utils import nbytes_from_nbits


class Scope(Enum):
    REGISTER = 0
    SHARED = 1


@dataclass(frozen=True, eq=False)
class Value:
    dtype: DataType
    shape: tuple[int, ...]

    def as_register_value(self) -> RegisterValue:
        assert isinstance(self, RegisterValue)
        return self

    def as_shared_value(self) -> SharedValue:
        assert isinstance(self, SharedValue)
        return self


@dataclass(frozen=True, eq=False)
class RegisterValue(Value):
    layout: Layout

    @property
    def size(self) -> int:
        return self.layout.local_size

    @staticmethod
    def create(dtype: DataType, layout: Layout) -> RegisterValue:
        return RegisterValue(dtype, layout.shape, layout)


@dataclass(frozen=True, eq=False)
class SharedValue(Value):
    layout: SharedLayout

    @staticmethod
    def create(dtype: DataType, layout: SharedLayout) -> SharedValue:
        return SharedValue(dtype, layout.shape, layout)

    @property
    def size(self) -> int:
        return self.layout.size

    def nbytes(self) -> int:
        return nbytes_from_nbits(self.size * self.dtype.nbits)
