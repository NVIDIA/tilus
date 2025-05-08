from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from hidet.ir.expr import Expr
from hidet.ir.type import DataType
from tilus.ir.layout import GlobalLayout, RegisterLayout, SharedLayout
from tilus.utils import nbytes_from_nbits


class Scope(Enum):
    REGISTER = 0
    SHARED = 1


@dataclass(frozen=True, eq=False)
class Tensor:
    dtype: DataType

    def as_register_tensor(self) -> RegisterTensor:
        assert isinstance(self, RegisterTensor)
        return self

    def as_shared_tensor(self) -> SharedTensor:
        assert isinstance(self, SharedTensor)
        return self

    def as_global_tensor(self) -> GlobalTensor:
        assert isinstance(self, GlobalTensor)
        return self

    def as_register_or_shared_tensor(self) -> RegisterTensor | SharedTensor:
        assert isinstance(self, (RegisterTensor, SharedTensor))
        return self


@dataclass(frozen=True, eq=False)
class RegisterTensor(Tensor):
    layout: RegisterLayout

    @staticmethod
    def create(dtype: DataType, layout: RegisterLayout) -> RegisterTensor:
        return RegisterTensor(dtype=dtype, layout=layout)

    @property
    def local_size(self) -> int:
        return self.layout.local_size

    @property
    def shape(self) -> tuple[int, ...]:
        return self.layout.shape

    """
    The following methods are used for type hinting in Tilus Script. The corresponding operations/methods will be
    converted in the Tilus Script transpiler defined in tilus.lang.transpiler module.
    """

    def __add__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        raise RuntimeError("tensor + tensor could only be used in Tilus Script.")

    def __sub__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        raise RuntimeError("tensor - tensor could only be used in Tilus Script.")

    def __mul__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        raise RuntimeError("tensor * tensor could only be used in Tilus Script.")

    def __truediv__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        raise RuntimeError("tensor / tensor could only be used in Tilus Script.")

    def squeeze(self, dim: int | Sequence[int]) -> RegisterTensor:
        raise RuntimeError("tensor.squeeze(...) could only be used in Tilus Script.")

    def unsqueeze(self, dim: int | Sequence[int]) -> RegisterTensor:
        raise RuntimeError("tensor.unsqueeze(...) could only be used in Tilus Script.")

    def transpose(self) -> RegisterTensor:
        raise RuntimeError("tensor.transpose(...) could only be used in Tilus Script.")


@dataclass(frozen=True, eq=False)
class SharedTensor(Tensor):
    layout: SharedLayout

    @staticmethod
    def create(dtype: DataType, layout: SharedLayout) -> SharedTensor:
        return SharedTensor(dtype=dtype, layout=layout)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.layout.shape

    @property
    def size(self) -> int:
        return self.layout.size

    @property
    def nbytes(self) -> int:
        return nbytes_from_nbits(self.size * self.dtype.nbits)

    def __getitem__(self, index: int | Expr) -> SharedTensor:
        raise RuntimeError("shared_tensor[...] could only be used in Tilus Script.")


@dataclass(frozen=True, eq=False)
class GlobalTensor(Tensor):
    layout: GlobalLayout

    @staticmethod
    def create(dtype: DataType, layout: GlobalLayout) -> GlobalTensor:
        return GlobalTensor(dtype=dtype, layout=layout)

    @property
    def shape(self) -> tuple[Expr, ...]:
        return self.layout.shape

    @property
    def size(self) -> Expr:
        return self.layout.size

    def __getitem__(self, indices: tuple[Expr | int, ...] | Expr | int) -> Expr:
        raise RuntimeError("global_tensor[...] could only be used in Tilus Script.")

    def with_layout(self, layout: GlobalLayout) -> GlobalTensor:
        return dataclasses.replace(self, layout=layout)
