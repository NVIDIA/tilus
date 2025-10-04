# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence

from hidet.ir.expr import Expr
from hidet.ir.type import DataType
from hidet.utils import same_list

from tilus.ir.layout import GlobalLayout, RegisterLayout, SharedLayout
from tilus.utils import nbytes_from_nbits


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

    def as_tmemory_tensor(self) -> TMemoryTensor:
        assert isinstance(self, TMemoryTensor)
        return self

    def as_global_tensor(self) -> GlobalTensor:
        assert isinstance(self, GlobalTensor)
        return self
    
    def as_shared_or_tmemory_tensor(self) -> SharedTensor | TMemoryTensor:
        assert isinstance(self, (SharedTensor, TMemoryTensor))
        return self

    def as_register_or_shared_tensor(self) -> RegisterTensor | SharedTensor:
        assert isinstance(self, (RegisterTensor, SharedTensor))
        return self


@dataclass(frozen=True, eq=False)
class RegisterTensor(Tensor):
    """A tensor that resides in the register memory.

    Attributes
    ----------
    dtype: DataType
        The data type of the tensor elements.
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
        if not isinstance(layout, RegisterLayout):
            raise ValueError(f"Layout must be a RegisterLayout, but got {type(layout)}.")
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

    __hash__ = object.__hash__  # use default hash function

    def __bool__(self):
        # We return True for all RegisterTensor so that we can use `if inst.output` to check whether the instruction
        # has an output tensor. We will handle the case of `if tensor:` in the Tilus Script transpiler differently to
        # avoid ambiguity.
        return True

    def __neg__(self):
        """Negate the tensor.

        Returns
        -------
        ret: RegisterTensor
            A new tensor that is the result of the negation.
        """
        raise RuntimeError(" -tensor could only be used in Tilus Script.")

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

    def __ge__(self, other):
        """Greater than or equal to comparison.

        Parameters
        ----------
        other: RegisterTensor | int | float | Expr
            The tensor or scalar to compare with this tensor.

        Returns
        -------
        ret: RegisterTensor
            A new tensor that is the result of the comparison.
        """
        raise RuntimeError("tensor >= tensor could only be used in Tilus Script.")

    def __le__(self, other):
        """Less than or equal to comparison.

        Parameters
        ----------
        other: RegisterTensor | int | float | Expr
            The tensor or scalar to compare with this tensor.

        Returns
        -------
        ret: RegisterTensor
            A new tensor that is the result of the comparison.
        """
        raise RuntimeError("tensor <= tensor could only be used in Tilus Script.")

    def __gt__(self, other):
        """Greater than comparison.

        Parameters
        ----------
        other: RegisterTensor | int | float | Expr
            The tensor or scalar to compare with this tensor.

        Returns
        -------
        ret: RegisterTensor
            A new tensor that is the result of the comparison.
        """
        raise RuntimeError("tensor > tensor could only be used in Tilus Script.")

    def __lt__(self, other):
        """Less than comparison.

        Parameters
        ----------
        other: RegisterTensor | int | float | Expr
            The tensor or scalar to compare with this tensor.

        Returns
        -------
        ret: RegisterTensor
            A new tensor that is the result of the comparison.
        """
        raise RuntimeError("tensor < tensor could only be used in Tilus Script.")

    def __eq__(self, other):
        """Equal to comparison.

        Parameters
        ----------
        other: RegisterTensor | int | float | Expr
            The tensor or scalar to compare with this tensor.

        Returns
        -------
        ret: RegisterTensor
            A new tensor that is the result of the comparison.
        """
        raise RuntimeError("tensor == tensor could only be used in Tilus Script.")

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
    """A tensor that resides in the shared memory.

    Attributes
    ----------
    dtype: DataType
        The data type of the tensor elements.
    shape: tuple[int, ...]
        The shape of the tensor.
    optional_layout: SharedLayout, optional
        The layout of the tensor, which is optional. When not provided, the layout will be automatically inferred
        with compiler pass.
    """

    shape: tuple[int, ...]
    optional_layout: Optional[SharedLayout]

    def __getitem__(self, indices: tuple[Expr | int, ...] | Expr | int) -> SharedTensor | Expr:
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
        """Get the layout of the SharedTensor.

        This property returns the layout of the SharedTensor. If the layout is not defined, it raises a ValueError.

        Returns
        -------
        ret: SharedLayout
            The layout of the SharedTensor.

        Raises
        ------
        ValueError
            If the SharedTensor does not have a layout defined.
        """
        if self.optional_layout is None:
            raise ValueError("SharedTensor does not have a layout defined.")
        return self.optional_layout

    @property
    def size(self) -> int:
        """Get the size of the SharedTensor.

        This property returns the storage size of the tensor as an expression, in the unit of number of elements.
        If the SharedTensor is not compact, it may not be equal to the product of the shape dimensions.

        Returns
        -------
        ret: int
            The size of the SharedTensor, which is the number of elements it contains.
        """
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
        if not isinstance(layout, SharedLayout):
            raise ValueError(f"Layout must be a SharedLayout, but got {type(layout)}.")
        if not same_list(self.shape, layout.shape):
            raise ValueError(f"Shape mismatch: provided shape {self.shape} does not match layout shape {layout.shape}.")
        return dataclasses.replace(self, optional_layout=layout)

    """
    The following methods are used for type hinting in Tilus Script. The corresponding operations/methods will be
    converted in the Tilus Script transpiler defined in tilus.lang.transpiler module.
    """

    def permute(self, dims: tuple[int, ...]) -> SharedTensor:
        raise RuntimeError("shared_tensor.permute(...) could only be used in Tilus Script.")


@dataclass(frozen=True, eq=False)
class TMemoryTensor(Tensor):
    shape: tuple[int, int]

    # the first lane of the tensor in tensor memory (0 to 127, inclusive)
    first_lane: int

    @staticmethod
    def create(dtype: DataType, shape: Sequence[int], first_lane: int) -> TMemoryTensor:
        if len(shape) != 2:
            raise ValueError("TensorMemoryTensor only supports 2D shape.")
        return TMemoryTensor(dtype=dtype, shape=(shape[0], shape[1]), first_lane=first_lane)


@dataclass(frozen=True, eq=False)
class GlobalTensor(Tensor):
    """A tensor that resides in the global memory.

    Attributes
    ----------
    dtype: DataType
        The data type of the tensor elements.

    layout: GlobalLayout
        The layout of the tensor, which defines how the tensor is stored in global memory.
    """

    layout: GlobalLayout

    @staticmethod
    def create(dtype: DataType, layout: GlobalLayout) -> GlobalTensor:
        return GlobalTensor(dtype=dtype, layout=layout)

    @property
    def shape(self) -> tuple[Expr, ...]:
        """Get the shape of the global tensor.

        This property returns the shape of the tensor as a tuple of expressions.

        Returns
        -------
        ret: tuple[Expr, ...]
            The shape of the global tensor.
        """
        return self.layout.shape

    @property
    def size(self) -> Expr:
        """Get the size of the global tensor.

        This property returns the storage size of the tensor as an expression, in the unit of number of elements.
        If the global tensor is not compact, it may not be equal to the product of the shape dimensions.

        Returns
        -------
        ret: Expr
            The size of the global tensor.
        """
        return self.layout.size

    def __getitem__(self, indices: tuple[Expr | int, ...] | Expr | int) -> Expr:
        """Access the global tensor with the given indices.

        This method allows you to access elements of the global tensor using indices.

        **This method is intended to be used in Tilus Script only.**

        Parameters
        ----------
        indices: tuple[Expr | int, ...] | Expr | int
            The indices to access the global tensor.


        Returns
        -------
        ret: Expr
            An expression representing the accessed element of the global tensor.
        """
        raise RuntimeError("global_tensor[...] could only be used in Tilus Script.")

    def with_layout(self, layout: GlobalLayout) -> GlobalTensor:
        if not isinstance(layout, GlobalLayout):
            raise ValueError(f"Layout must be a GlobalLayout, but got {type(layout)}.")
        return dataclasses.replace(self, layout=layout)
