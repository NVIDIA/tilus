from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence

from hidet.ir.expr import Expr
from hidet.ir.type import DataType
from hidet.utils import same_list

from tilus.ir.layout import GlobalLayout, RegisterLayout, SharedLayout
from tilus.utils import nbytes_from_nbits, prod


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
    shape: tuple[int, ...]
    optional_layout: Optional[RegisterLayout] = None

    @staticmethod
    def create(
        dtype: DataType, *, shape: Optional[Sequence[int]] = None, optional_layout: Optional[RegisterLayout] = None
    ) -> RegisterTensor:
        if shape is None and optional_layout is None:
            raise ValueError("Either shape or layout must be provided to create a RegisterTensor.")
        elif shape is None:
            shape = optional_layout.shape
        elif optional_layout is None:
            pass  # layout is optional
        else:
            if len(shape) != len(optional_layout.shape) or not same_list(shape, optional_layout.shape):
                raise ValueError(
                    f"Shape mismatch: provided shape {shape} does not match layout shape {optional_layout.shape}."
                )
        return RegisterTensor(dtype=dtype, shape=tuple(shape), optional_layout=optional_layout)

    @cached_property
    def layout(self) -> RegisterLayout:
        if self.optional_layout is None:
            raise ValueError("The layout of RegisterTensor is not defined yet.")
        return self.optional_layout

    @cached_property
    def size(self) -> int:
        return prod(self.shape)

    @cached_property
    def local_size(self) -> int:
        return self.layout.local_size

    def with_layout(self, layout: RegisterLayout) -> RegisterTensor:
        """
        Create a new RegisterTensor with the given layout.
        """
        if not same_list(self.shape, layout.shape):
            raise ValueError(f"Shape mismatch: provided shape {self.shape} does not match layout shape {layout.shape}.")
        return dataclasses.replace(self, optional_layout=layout)

    def has_layout(self) -> bool:
        """
        Check if the RegisterTensor has a layout defined.
        """
        return self.optional_layout is not None

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
    shape: tuple[int, ...]
    optional_layout: Optional[SharedLayout]

    def __getitem__(self, index: int | Expr) -> SharedTensor:
        raise RuntimeError("shared_tensor[...] could only be used in Tilus Script.")

    @staticmethod
    def create(
        dtype: DataType, *, shape: Sequence[int] = None, optional_layout: Optional[SharedLayout] = None
    ) -> SharedTensor:
        if shape is None and optional_layout is None:
            raise ValueError("Either shape or layout must be provided to create a SharedTensor.")
        elif shape is None:
            shape = optional_layout.shape
        elif optional_layout is None:
            pass  # layout is optional
        else:
            if len(shape) != len(optional_layout.shape) or not same_list(shape, optional_layout.shape):
                raise ValueError(
                    f"Shape mismatch: provided shape {shape} does not match layout shape {optional_layout.shape}."
                )
        return SharedTensor(dtype=dtype, shape=tuple(shape), optional_layout=optional_layout)

    @property
    def layout(self) -> SharedLayout:
        if self.optional_layout is None:
            raise ValueError("SharedTensor does not have a layout defined.")
        return self.optional_layout

    @property
    def size(self) -> int:
        return self.layout.size

    @property
    def nbytes(self) -> int:
        return nbytes_from_nbits(self.size * self.dtype.nbits)

    def has_layout(self) -> bool:
        return self.optional_layout is not None

    def with_layout(self, layout: SharedLayout) -> SharedTensor:
        """
        Create a new SharedTensor with the given layout.
        """
        if not same_list(self.shape, layout.shape):
            raise ValueError(f"Shape mismatch: provided shape {self.shape} does not match layout shape {layout.shape}.")
        return dataclasses.replace(self, optional_layout=layout)


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
