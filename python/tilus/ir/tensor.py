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
    """Base class for all tensor types in Tilus.

    Attributes
    ----------
    dtype: DataType
        The data type of the tensor elements.
    """

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
    """A tensor that resides in the register memory.

    Attributes
    ----------
    shape: tuple[int, ...]
        The shape of the tensor.
    optional_layout: Optional[RegisterLayout]
        The layout of the tensor, which is optional. When not provided, the layout will be automatically inferred
        with compiler pass.
    """

    shape: tuple[int, ...]
    optional_layout: Optional[RegisterLayout] = None

    @staticmethod
    def create(
        dtype: DataType, *, shape: Sequence[int], optional_layout: Optional[RegisterLayout] = None
    ) -> RegisterTensor:
        """
        Create a RegisterTensor with the given dtype, shape, and optional layout.

        Parameters
        ----------
        dtype: DataType
            The data type of the tensor elements.
        shape: Sequence[int]
            The shape of the tensor.
        optional_layout: RegisterLayout, optional
            The layout of the tensor. If not provided, the layout will be inferred later.

        Returns
        -------
        ret: RegisterTensor
            The created RegisterTensor instance.
        """
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
        """Get the layout of the RegisterTensor.

        Returns
        -------
        ret: RegisterLayout
            The layout of the RegisterTensor.

        Raises
        ------
        ValueError
            If the layout of the RegisterTensor is not defined yet.
        """
        if self.optional_layout is None:
            raise ValueError("The layout of RegisterTensor is not defined yet.")
        return self.optional_layout

    @cached_property
    def size(self) -> int:
        """Get the number of elements in the RegisterTensor.

        Returns
        -------
        ret: int
            The number of elements in the RegisterTensor.
        """
        return prod(self.shape)

    @cached_property
    def local_size(self) -> int:
        """Get the number of elements stored in each thread.

        Returns
        -------
        ret: int
            The number of elements stored in each thread.
        """
        return self.layout.local_size

    def with_layout(self, layout: RegisterLayout) -> RegisterTensor:
        """Create a new RegisterTensor with the given layout.

        Parameters
        ----------
        layout: RegisterLayout
            The layout to be used for the new RegisterTensor.

        Returns
        -------
        ret: RegisterTensor
            A new RegisterTensor instance with the specified layout.
        """
        if not same_list(self.shape, layout.shape):
            raise ValueError(f"Shape mismatch: provided shape {self.shape} does not match layout shape {layout.shape}.")
        return dataclasses.replace(self, optional_layout=layout)

    def has_layout(self) -> bool:
        """Check if the RegisterTensor has a layout defined.

        Returns
        -------
        ret: bool
            True if the RegisterTensor has a layout defined, False otherwise.
        """
        return self.optional_layout is not None

    """
    The following methods are used for type hinting in Tilus Script. The corresponding operations/methods will be
    converted in the Tilus Script transpiler defined in tilus.lang.transpiler module.
    """

    def __add__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        """Perform addition with another tensor or a scalar.

        Parameters
        ----------
        other: RegisterTensor | int | float | Expr
            The tensor or scalar to add to this tensor.

        Returns
        -------
        ret: RegisterTensor
            A new tensor that is the result of the addition.
        """
        raise RuntimeError("tensor + tensor could only be used in Tilus Script.")

    def __sub__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        """Perform subtraction with another tensor or a scalar.

        Parameters
        ----------
        other: RegisterTensor | int | float | Expr
            The tensor or scalar to subtract from this tensor.

        Returns
        -------
        ret: RegisterTensor
            A new tensor that is the result of the subtraction.

        """
        raise RuntimeError("tensor - tensor could only be used in Tilus Script.")

    def __mul__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        """Perform multiplication with another tensor or a scalar.

        Parameters
        ----------
        other: RegisterTensor | int | float | Expr
            The tensor or scalar to multiply with this tensor.

        Returns
        -------
        ret: RegisterTensor
            A new tensor that is the result of the multiplication.
        """
        raise RuntimeError("tensor * tensor could only be used in Tilus Script.")

    def __truediv__(self, other: RegisterTensor | int | float | Expr) -> RegisterTensor:
        """Perform division with another tensor or a scalar.

        Parameters
        ----------
        other: RegisterTensor | int | float | Expr
            The tensor or scalar to divide this tensor by.

        Returns
        -------
        ret: RegisterTensor
            A new tensor that is the result of the division.
        """
        raise RuntimeError("tensor / tensor could only be used in Tilus Script.")

    def squeeze(self, dim: int | Sequence[int]) -> RegisterTensor:
        """Squeeze the tensor by removing dimensions of size 1.

        Parameters
        ----------
        dim: int | Sequence[int]
            The dimension(s) to squeeze. If an integer is provided, it will squeeze that specific dimension.
            If a sequence of integers is provided, it will squeeze all specified dimensions.

        Returns
        -------
        ret: RegisterTensor
            A new tensor with the specified dimensions squeezed.

        See Also
        --------
        :py:func:`~tilus.Script.squeeze`
        """
        raise RuntimeError("tensor.squeeze(...) could only be used in Tilus Script.")

    def unsqueeze(self, dim: int | Sequence[int]) -> RegisterTensor:
        raise RuntimeError("tensor.unsqueeze(...) could only be used in Tilus Script.")

    def transpose(self) -> RegisterTensor:
        raise RuntimeError("tensor.transpose(...) could only be used in Tilus Script.")

    def to(self, dtype: DataType) -> RegisterTensor:
        raise RuntimeError("tensor.to(...) could only be used in Tilus Script.")


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
