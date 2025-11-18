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
import typing
from typing import Callable, Optional, Sequence, Union

from hidet.ir.dtypes import boolean
from hidet.ir.expr import Expr, Var, as_expr
from hidet.ir.tools import infer_type
from hidet.ir.type import DataType

from tilus.ir.inst import InstructionError
from tilus.ir.layout import GlobalLayout, RegisterLayout, SharedLayout
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor, Tensor

from .base import InstructionGroup


class RootInstructionGroup(InstructionGroup):
    def assume(self, cond: Expr | bool) -> None:
        """Compiler hint to assume a condition is true.

        This method is used to provide a condition that the compiler can assume to be true. It is typically used
        to provide additional information to the compiler for optimization purposes.

        The condition can be a boolean expression with the following forms:

        - term
        - term [and term]*

        where `term` can be one of the following forms:

        - a % c == 0, where `a` is a kernel parameter and `c` is a constant.

        Parameters
        ----------
        cond: Expr | bool
            The condition to assume. It must be an expression that evaluates to a boolean value or a boolean value.

        Raises
        ------
        InstructionError
            If the condition is not a boolean expression or if it cannot be recognized.
        """
        if isinstance(cond, bool):
            if not cond:
                raise InstructionError("The condition must be True")
            return
        assert isinstance(cond, Expr), "The condition must be a boolean expression"
        self._builder.assume(cond)

    def register_tensor(
        self,
        *,
        dtype: DataType,
        shape: Sequence[int],
        init: Optional[Callable[[Var, ...], Expr | int | float | bool] | Expr | int | float] = None,  # type: ignore [misc]
    ) -> RegisterTensor:
        """Create a register tensor.

        This instruction allocates a register tensor with the specified data type, shape, (optional) layout, and
        (optional) initialization value.

        If `init` is not provided, the register tensor will be uninitialized. If `init` is provided, it can be a
        scalar value (e.g., `int`, `float`, `bool`, or a scalar expression) that will be used to initialize all elements
        of the register tensor. It can also be a callable function that takes a sequence of index variables (i, j, ...)
        and returns a scalar expression based on these indices. The element at index (i, j, ...) will be initialized
        with the value returned by this function. When the data type of the value is not identical to the data type of
        the register tensor, the value will be cast to the data type of the register tensor.

        Parameters
        ----------
        dtype: DataType
            The data type of the tensor elements.
        shape: Sequence[int]
            The shape of the tensor.
        init: Callable[[Var, ...], Expr | int | float | bool] | Expr | int | float, optional
            The initialization value or function to initialize the tensor elements.

        Returns
        -------
        tensor: RegisterTensor
            The allocated register tensor.
        """

        f_init: Optional[Callable[[Sequence[Var]], Expr]] = None
        if init is not None:

            def f_init_(indices: Sequence[Var]) -> Expr:
                if isinstance(init, (float, int, bool, Expr)):
                    return dtype.constant(init)  # noqa: E731
                elif callable(init):
                    return init(*indices)  # type: ignore
                else:
                    raise ValueError("init must be a callable, int, float, bool, or Expr, got {}".format(type(init)))

            f_init = f_init_

        return self._builder.allocate_register(dtype=dtype, shape=shape, f_init=f_init)

    def global_tensor(
        self,
        dtype: DataType,
        shape: Sequence[int | Expr],
        *,
        layout: Optional[GlobalLayout] = None,
        requires_clean: bool,
    ) -> GlobalTensor:
        """Allocate a global tensor.

        This instruction allocates a global tensor with the specified data type, shape, layout, and whether it requires
        to be all zeros. All thread blocks in the kernel must agree on the shape and layout of the global tensor. The
        global tensor will be shared across all thread blocks in the kernel. The lifetime of the global tensor is
        the entire kernel execution, and it will be automatically freed when the kernel finishes.

        The `requires_clean` parameter indicates whether the global tensor should be initialized to all zeros.

        - If it is set to `True`, the global tensor will be initialized to all zeros. We require the kernel to reset
          the global tensor to all zeros after the kernel finishes.
        - If it is set to `False`, the global tensor will be uninitialized, and its contents are undefined.

        Parameters
        ----------
        dtype: DataType
            The data type of the tensor elements.
        shape: Sequence[int | Expr]
            The shape of the tensor. The shape can be a sequence of integers or integer expressions.
        layout: GlobalLayout, optional
            The layout of the tensor. If not provided, the layout will be row-major compact layout by default.
        requires_clean: bool
            Whether the global tensor should be initialized to all zeros.

        Returns
        -------
        ret: GlobalTensor
            The allocated global tensor.
        """
        return self._builder.allocate_global(
            dtype=dtype,
            shape=shape,
            layout=layout,
            requires_clean=requires_clean,
        )

    def shared_tensor(
        self,
        *,
        dtype: DataType,
        shape: Sequence[int],
    ) -> SharedTensor:
        """Allocate a shared tensor.

        This instruction allocates a shared tensor with the specified data type, shape, and (optional) layout.

        Parameters
        ----------
        dtype: DataType
            The data type of the tensor elements.
        shape: Sequence[int]
            The shape of the tensor.

        Returns
        -------
        ret: SharedTensor
            The allocated shared tensor.
        """
        return self._builder.allocate_shared(dtype=dtype, shape=shape)

    def global_view(
        self,
        ptr: Expr,
        *,
        dtype: DataType,
        shape: Sequence[Expr | int],
        strides: Optional[Sequence[Expr | int]] = None,
    ) -> GlobalTensor:
        """Create a global tensor view.

        There are three ways to specify the layout:

        - `layout`: If provided, it overrides the shape and strides parameters.
        - `shape`: If provided, it defines the shape of the tensor and assume a compact row-major strides.
        - `shape` and `strides`: If provided, they define the shape and strides of the tensor.

        Parameters
        ----------
        ptr: Expr
            The pointer to the global memory, which should be a pointer expression to the first element of the tensor.
        dtype: DataType
            The data type of the tensor elements.
        shape: Sequence[Expr | int]
            The shape of the tensor.
        strides: Sequence[Expr | int], optional
            The strides of the tensor. If not provided, it is assumed to be compact row-major layout.
        layout: GlobalLayout, optional
            The layout of the tensor. If provided, it overrides the shape and strides parameters.

        Returns
        -------
        ret: GlobalTensor
            The global tensor view created.
        """
        from tilus.ir.layout import global_row_major, global_strides

        assert shape is not None, "Must specify shape when layout is not provided"
        if strides is None:
            # assume compact row-major layout
            layout = global_row_major(*shape)
        else:
            assert len(shape) == len(strides), "Shape and strides must have the same length"
            layout = global_strides(shape, strides)

        return self._builder.global_view(ptr=ptr, dtype=dtype, layout=layout)

    def load_global(
        self,
        src: GlobalTensor,
        /,
        *,
        offsets: Sequence[Expr | int],
        shape: Sequence[int],
        dims: Optional[Sequence[int]] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Load a slice of global tensor into a register tensor.

        This instruction loads a slice of the global tensor `x` into a register tensor, given the `offsets` for each
        dimension of the global tensor and the `shape` of the slice to be loaded.

        When we only slice over a subset of the dimensions of the global tensor, we can specify the `dims` parameter to
        indicate which dimensions are being sliced.

        When `out` is provided, the loaded data will be stored in the `out` register tensor, otherwise a new register
        tensor will be allocated.

        Parameters
        ----------
        src: GlobalTensor
            The global tensor to load from.
        offsets: Sequence[Expr | int]
            The offsets for each dimension of the global tensor. The length of this sequence must match the number
            of dimensions of the global tensor.
        shape: Sequence[int], optional
            The shape of the slice to be loaded. If not provided, the shape of the global tensor will be used.
        dims: Sequence[int], optional
            The dimensions of the global tensor that are being sliced. If not provided, it is assumed that all
            dimensions are being sliced. The length of this sequence must match the number of dimensions of the
            register tensor being loaded into.
        out: RegisterTensor, optional
            The register tensor to store the loaded data into. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the loaded data from the global tensor.
        """
        if len(offsets) != len(src.shape):
            raise InstructionError(
                "The number of offsets must be equal to the number of dimensions of the global tensor"
            )
        return self._builder.load_global(x=src, offsets=offsets, dims=dims, shape=shape, output=out)

    def store_global(
        self,
        dst: GlobalTensor,
        src: RegisterTensor,
        *,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
    ) -> None:
        """Store a register tensor into a slice of a global tensor.

        This instruction stores the contents of the register tensor `x` into a slice of the global tensor `dst`.

        The `offsets` parameter specifies the starting offsets for each dimension of the global tensor where the
        register tensor will be stored. The length of this sequence must match the number of dimensions of the global
        tensor.

        The `dims` parameter specifies which dimensions of the global tensor are being sliced. The dimension dim[0] of
        the global tensor corresponds to the first dimension of the register tensor, dim[1] to the second, and so on.
        If `dims` is not provided, it is assumed to be range(len(dst.shape)), meaning all dimensions of the global tensor
        are being sliced in the same order as the register tensor. When provided, the length of this sequence must
        match the number of dimensions of the register tensor being stored.

        Parameters
        ----------
        dst: GlobalTensor
            The global tensor to store into.
        src: RegisterTensor
            The register tensor to store into the global tensor.
        offsets: Sequence[Expr | int]
            The offsets for each dimension of the global tensor where the register tensor will be stored.
        dims: Sequence[int], optional
            The dimensions of the global tensor that are being sliced.
        """
        if dims is not None and len(dims) != len(src.shape):
            raise InstructionError(
                "The number of slice dimensions must be equal to the number of dimensions of the "
                f"register tensor: {len(dims)} vs {len(src.shape)}"
            )
        return self._builder.store_global(dst=dst, src=src, offsets=offsets, dims=dims)

    def load_shared(
        self,
        src: SharedTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Load a shared tensor into a register tensor.

        Parameters
        ----------
        src: SharedTensor
            The shared tensor to load from.
        out: RegisterTensor, optional
            The register tensor to store the loaded data into. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the loaded data from the shared tensor.
        """
        return self._builder.load_shared(src=src, output=out)

    def store_shared(
        self,
        dst: SharedTensor,
        src: RegisterTensor,
        *,
        offsets: Optional[Sequence[int]] = None,
        dims: Optional[Sequence[int]] = None,
    ) -> None:
        """Store a register tensor into a shared tensor.

        This instruction stores the contents of the register tensor `src` into a slice of the shared tensor `dst`.

        Parameters
        ----------
        dst: SharedTensor
            The shared tensor to store into.
        src: RegisterTensor
            The register tensor to store into the shared tensor.
        offsets: Sequence[int], optional
            The offsets for each dimension of the shared tensor where the register tensor will be stored.
        dims: Sequence[int], optional
            The dimensions of the shared tensor that are being sliced. If not provided, it is assumed that all
            dimensions are being sliced in the same order as the register tensor. The length of this sequence must
            match the number of dimensions of the register tensor being stored.
        """
        if dst.dtype != src.dtype:
            raise InstructionError(
                "Cannot store shared tensor {}{} from register tensor {}{}: dtype mismatch".format(
                    dst.dtype.name, list(dst.shape), src.dtype.name, list(src.shape)
                )
            )
        if offsets is not None:
            assert len(offsets) == len(dst.shape)
            if dims is None:
                assert len(src.shape) == len(dst.shape)
                dims = list(range(len(src.shape)))
            dst = self._builder.slice_shared(dst, offsets=offsets, slice_dims=dims, slice_shape=src.shape)
        self._builder.store_shared(dst=dst, src=src)

    def free_shared(self, tensor: SharedTensor) -> None:
        """Free a shared tensor.

        Parameters
        ----------
        tensor: SharedTensor
            The shared tensor to free.
        """
        self._builder.free_shared(tensor)

    def copy_async(
        self,
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
        evict: Optional[str] = None,
        check_bounds: bool = True,
    ) -> None:
        """Copy from global to shared tensor asynchronously.

        This instruction issues an asynchronous  copy of a tile from a global tensor to a shared tensor. The `src` parameter
        specifies the global tensor to copy from, while the `dst` parameter specifies the shared tensor to copy to.

        The `offsets` parameter specifies the starting offsets for each dimension of the global tensor where the tile
        will be copied from. The length of this sequence must match the rank of the global tensor.

        The `dims` parameter specifies which dimensions of the global tensor are being sliced. If not provided, it is
        assumed that all dimensions are being sliced in the same order as the shared tensor. The length of this sequence
        must match the number of dimensions of the shared tensor being copied to.

        The `evict` parameter can be used to specify the eviction policy. When we use this instruction, the data in
        the global memory will be cached. We can use the `evict` parameter to specify the eviction policy for the cached
        data of this instruction.

        It's valid to specify the loading elements out of bounds of the global tensor, in which case, we will perform
        bound checking and fill the out-of-bounds elements with zero in the shared tensor. The bound checking might
        introduce some overhead, especially when the user make sure that the accessed global elements are always in
        bounds but our compiler cannot infer it. In this case, we can set `check_bounds` to `False` to skip the bound
        checking. It's the user's responsibility to ensure that the accessed global elements are always in bounds when
        `check_bounds` is set to `False`.

        Parameters
        ----------
        src: GlobalTensor
            The global tensor to copy from.
        dst: SharedTensor
            The shared tensor to copy to.
        offsets: Sequence[Expr | int]
            The offsets for each dimension of the global tensor where the tile will be copied from. The length of this
            sequence must match the number of dimensions of the global tensor.
        dims: Sequence[int], optional
            The dimensions of the global tensor that are being sliced when the rank of shared tensor is less than the
            rank of the global tensor. If not provided, it is assumed that all dimensions are being sliced in the
            same order as the shared tensor. The length of this sequence must match the number of dimensions of the
            shared tensor being copied to.
        evict: str, optional
            The eviction policy for the cached data of this instruction. If not provided, the default eviction
            policy `evict_normal` is used, which is to evict the cached data when the shared memory is full. The
            eviction policy can be one of

            The candidates are:

            - 'evict_normal': Evict the cached data when the shared memory is full.
            - 'evict_first': Evict the cached data of this instruction first when an eviction is needed. This policy is
              suitable for streaming data where the data is only needed once and will not be reused.

        check_bounds: bool, optional
            Whether to check the bounds of the accessed global elements. When set to `True`, the accessed global
            elements will be checked to ensure they are within bounds. If any accessed global element is out of bounds,
            it will be filled with zero in the shared tensor. When set to `False`, the bound checking will be skipped,
            and the user must ensure that the accessed global elements are always in bounds. The default value is `True`.
        """
        if dims is None:
            if len(dst.shape) != len(src.shape):
                raise InstructionError(
                    "The number of dimensions of the source global tensor must match the destination shared tensor if dims is not specified"
                )
        if len(offsets) != len(src.shape):
            raise InstructionError(
                "The number of offsets must be equal to the number of dimensions of the source global tensor"
            )
        self._builder.copy_async(dst=dst, src=src, offsets=offsets, dims=dims, evict=evict, check_bounds=check_bounds)

    def copy_async_wait_all(self):
        """Wait for all copy_async instructions to complete.

        This instruction is equivalent to:

        .. code-block:: python

            self.copy_async_commit_group()
            self.copy_async_wait_group(0)

        """
        self._builder.copy_async_wait_all()

    def copy_async_commit_group(self):
        """Commit async copies into a group.

        This instruction commits all the pending asynchronous copy operations into a group.

        """
        self._builder.copy_async_commit_group()

    def copy_async_wait_group(self, n: Union[Expr, int]) -> None:
        """Wait the completion of asynchronous copy groups.

        This instruction waits for the completion of asynchronous copy groups. The `n` parameter specifies the maximum
        number of asynchronous copy groups that can be unfinished at the same time. If `n` is 0, it will wait until all
        asynchronous copy groups are finished. If `n` is greater than 0, it will wait until at least `n` asynchronous
        copy groups are finished, allowing up to `n` asynchronous copy groups to be unfinished at the same time.

        Parameters
        ----------
        n: Union[Expr, int]
            The maximum number of asynchronous copy groups that can be unfinished at the same time. If `n` is 0,
            it will wait until all asynchronous copy groups are finished.
        """
        self._builder.copy_async_wait_group(n)

    def dot(
        self,
        a: RegisterTensor,
        b: RegisterTensor,
        c: Optional[RegisterTensor] = None,
        /,
        *,
        acc_dtype: Optional[DataType] = None,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Dot product.

        This instruction computes the dot product: `out = a @ b + c`.

        The `a`, `b` and (optional) `c` tensors must be 2D register tensors, where

        - `a` has shape [m, k]
        - `b` has shape [k, n]
        - `c` has shape [m, n]

        If `c` is not provided, it's assumed to be a zero-initialized accumulator tensor with `acc_dtype` as its data
        type. If both `c` and `acc_dtype` are not provided, an error is raised.

        The `out` tensor is optional. If provided, it will be used to store the result of the dot product. If not
        provided, a new register tensor will be allocated to hold the result.

        The data type of the `c` and `out` must be the same and match the `acc_dtype` if they are provided.

        Parameters
        ----------
        a: RegisterTensor
            The first input tensor with shape [m, k].
        b: RegisterTensor
            The second input tensor with shape [k, n].
        c: RegisterTensor, optional
            The accumulator tensor with shape [m, n]. If not provided, a zero-initialized tensor will be used.
        acc_dtype: DataType, optional
            The data type of the accumulation computation. If `c` is not provided, this is used to determine the
            data type of the `c` tensor. If `c` is provided, it must match the data type of `c`.
        out: RegisterTensor, optional
            The output tensor to store the result of the dot product. If not provided, a new register tensor will be
            allocated to hold the result.

        Returns
        -------
        ret: RegisterTensor
            The result of the dot product, which is a register tensor with shape [m, n]. It will be `out` if provided,
            or a new register tensor if not.
        """
        if c is None:
            if acc_dtype is None:
                raise InstructionError('mma_dot requires either "c" or "acc_dtype" to be specified')
            m, n = a.shape[-2], b.shape[-1]
            c = self._builder.allocate_register(
                dtype=acc_dtype,
                shape=[m, n],
                f_init=lambda _: acc_dtype.constant(0),
            )
        else:
            if acc_dtype is not None and acc_dtype != c.dtype:
                raise InstructionError(
                    "The dtype of the accumulator tensor 'c' must match the specified 'acc_dtype' if provided"
                )
        if not (len(a.shape) == len(b.shape) == len(c.shape) == 2):
            raise InstructionError("mma_dot requires 2D tensors for a, b, and c")
        if a.shape[1] != b.shape[0] or a.shape[0] != c.shape[0] or b.shape[1] != c.shape[1]:
            raise InstructionError(
                "The shapes of a, b, and c must match for dot: "
                f"a: {a.shape}, b: {b.shape}, c: {c.shape} (expected a.shape[1] == b.shape[0] and a.shape[0] == c.shape[0] and b.shape[1] == c.shape[1])"
            )
        return self._builder.dot(
            a,
            b,
            c,
            output=out,
        )

    def cast(self, x: RegisterTensor, dtype: DataType) -> RegisterTensor:
        """Cast a register tensor to a different data type.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to be cast.
        dtype: DataType
            The target data type to cast the register tensor to.

        Returns
        -------
        ret: RegisterTensor
            The register tensor with the specified data type.
        """
        return self._builder.cast(x=x, dtype=dtype)

    def view(
        self,
        x: RegisterTensor,
        *,
        layout: Optional[RegisterLayout] = None,
        dtype: Optional[DataType] = None,
    ) -> RegisterTensor:
        """View register tensor with a different layout or data type.

        This instruction allows you to reinterpret a register tensor with a different layout or data type without
        changing its underlying data.

        The `layout` parameter specifies the new layout of the register tensor, while the `dtype` parameter specifies
        the new data type.

        There is a requirement for the `layout` and `dtype` parameters:

          x.dtype.nbits * x.layout.local_size == dtype.nbits * layout.local_size

        This means that the total number of bits stored in each thread must remain the same after reinterpretation.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to reinterpret.
        layout: RegisterLayout, optional
            The new layout of the register tensor. If not provided, the layout will remain unchanged.
        dtype: DataType, optional
            The new data type of the register tensor. If not provided, the data type will remain unchanged.

        Returns
        -------
        ret: RegisterTensor
            The register tensor with the specified layout and/or data type.
        """
        return self._builder.view(x=x, layout=layout, dtype=dtype, local_offset=0)

    def squeeze(
        self,
        x: RegisterTensor,
        *,
        dim: int | Sequence[int],
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Squeeze a dimension of a register tensor with size 1.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to squeeze.
        dim: int | Sequence[int]
            The dimension(s) to squeeze out. The dimension(s) must have size 1.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor with the specified dimension squeezed out.
        """
        return self._builder.squeeze(x, dim=dim, out=out)

    def unsqueeze(
        self,
        x: RegisterTensor,
        *,
        dim: int | Sequence[int],
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Unsqueeze a dimension of a register tensor.

        This instruction adds a new dimension of size 1 to the register tensor at the specified position. The
        `dim` parameter is the position where the new dimension will be added in the output tensor.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to unsqueeze.
        dim: int | Sequence[int]
            The dimension(s) to unsqueeze. If a single integer is provided, it specifies the position of the new
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor with the specified dimension(s) unsqueezed.
        """
        return self._builder.unsqueeze(x, dim=dim, out=out)

    def transpose(
        self,
        x: RegisterTensor,
    ) -> RegisterTensor:
        """Transpose a 2-D register tensor.

        This instruction transposes a 2-D register tensor, swapping its first and second dimensions.
        This instruction does not change the underlying data of the tensor, but only create a tensor with a new layout.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to transpose. It must be a 2-D tensor.

        Returns
        -------
        ret: RegisterTensor
            The transposed register tensor. The shape of the output tensor will be [x.shape[1], x.shape[0]].
        """
        return self._builder.transpose(x)

    def abs(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute the absolute value of a register tensor.

        This instruction computes the absolute value of each element in the register tensor `x`. The result is a new
        register tensor with the same dtype, shape, and layout as `x`.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to compute the absolute value of.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the absolute values of the elements in `x`. The shape and dtype of the
            output tensor will be the same as that of `x`.
        """
        return self._builder.abs(x, out=out)

    def exp(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute the exponential of each element.

        This instruction computes the natural exponential of each element in the register tensor `x`. The result is a
        new register tensor with the same dtype, shape, and layout as `x`.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to compute the exponential of.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the exponential values of the elements in `x`. The shape and dtype of the
            output tensor will be the same as that of `x`.
        """
        return self._builder.exp(x, out=out)

    def exp2(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute the base-2 exponential of each element.

        This instruction computes the base-2 exponential of each element in the register tensor `x`. The result is a
        new register tensor with the same dtype, shape, and layout as `x`.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to compute the base-2 exponential of.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the base-2 exponential values of the elements in `x`. The shape and dtype of the
            output tensor will be the same as that of `x`.
        """
        return self._builder.exp2(x, out=out)

    def log(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute the natural logarithm of each element.

        This instruction computes the natural logarithm of each element in the register tensor `x`. The result is a
        new register tensor with the same dtype, shape, and layout as `x`.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to compute the natural logarithm of.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the natural logarithm values of the elements in `x`. The shape and dtype of the
            output tensor will be the same as that of `x`.
        """
        return self._builder.log(x, out=out)

    def round(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Round each element to the nearest integer.

        This instruction rounds each element in the register tensor `x` to the nearest integer. The result is a new
        register tensor with the same dtype, shape, and layout as `x`. We use the "round-to-nearest-even" rounding mode.
        This means that if the fractional part of a number is exactly 0.5, it will be rounded to the nearest even integer.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to round.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the rounded values of the elements in `x`. The shape and dtype of the
            output tensor will be the same as that of `x`.
        """
        return self._builder.round(x, out=out)

    def sqrt(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute the square root of each element.

        This instruction computes the square root of each element in the register tensor `x`. The result is a new
        register tensor with the same dtype, shape, and layout as `x`.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to compute the square root of.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the square root values of the elements in `x`. The shape and dtype of the
            output tensor will be the same as that of `x`.
        """
        return self._builder.sqrt(x, out=out)

    def clip(
        self,
        x: RegisterTensor,
        min: Expr | int | float,
        max: Expr | int | float,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Clip the values of a register tensor to a specified range.

        This instruction clips the values of the register tensor `x` to the range specified by `min` and `max`. The
        resulting values will be in the range [min, max]. The result is a new register tensor with the same
        dtype, shape, and layout as `x`.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to clip.
        min: Expr | int | float
            The minimum value to clip the elements of `x` to. Any value less than `min` will be set to `min`.
        max: Expr | int | float
            The maximum value to clip the elements of `x` to. Any value greater than `max` will be set to `max`.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the clipped values of the elements in `x`. The shape and dtype of the
            output tensor will be the same as that of `x`.
        """
        return self._builder.clip(x=x, min=min, max=max, out=out)

    def repeat(
        self,
        x: RegisterTensor,
        repeats: Sequence[int],
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Repeat elements of a register tensor along its dimensions.

        This instruction repeats the elements of the register tensor `x` along each dimension according to the
        `repeats` parameter. The `repeats` parameter is a sequence of integers, where each integer specifies how many
        times to repeat the elements along the corresponding dimension of `x`.

        The difference between :py:meth:`repeat` and :py:meth:`repeat_interleave`
        is similar to the `torch.Tensor.repeat` function vs. `torch.Tensor.repeat_interleave`.

        Use one dimension tensor as an example:

        .. code-block:: python

           a = [1, 2, 3]
           repeat(a, [2])  # Output: [1, 2, 3, 1, 2, 3]
           repeat_interleave(a, [2])  # Output: [1, 1, 2, 2, 3, 3]

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to repeat.
        repeats: Sequence[int]
            The number of times to repeat the elements along each dimension of `x`. If the length of `repeats` is less
            than the number of dimensions of `x`, it will be padded with 1s for the beginning dimensions. If it is
            longer, we will expand the `x` tensor with 1s in the beginning dimensions to match the length of
            `repeats`.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the repeated elements of `x`. The shape of the output tensor will be
            determined by the `repeats` parameter, and its dtype will be the same as that of `x`.

        See Also
        --------
        :py:meth:`torch.Tensor.repeat`: A similar function in PyTorch.
        """
        return self._builder.repeat(
            x=x,
            repeats=repeats,
            out=out,
        )

    def repeat_interleave(
        self,
        x: RegisterTensor,
        repeats: Sequence[int],
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Repeat elements of a register tensor along its dimensions.

        This instruction repeats each element of the register tensor `x` according to the `repeats` parameter. The
        `repeats` parameter is a sequence of integers, where each integer specifies how many times to repeat the
        corresponding element of `x`.

        The difference between :py:meth:`repeat` and :py:meth:`repeat_interleave`
        is similar to the `torch.Tensor.repeat` function vs. `torch.Tensor.repeat_interleave`.

        Use one dimension tensor as an example:

        .. code-block:: python

           a = [1, 2, 3]
           repeat(a, [2])  # Output: [1, 2, 3, 1, 2, 3]
           repeat_interleave(a, [2])  # Output: [1, 1, 2, 2, 3, 3]

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to repeat.
        repeats: Sequence[int]
            The number of times to repeat each element of `x`. If the length of `repeats` is less than the number
            of dimensions of `x`, it will be padded with 1s for the beginning dimensions. If it is longer, we will
            expand the `x` tensor with 1s in the beginning dimensions to match the length of `repeats`.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the repeated elements of `x`. The shape of the output tensor will be
            determined by the `repeats` parameter, and its dtype will be the same as that of `x`.
        """
        return self._builder.repeat_interleave(
            x=x,
            repeats=repeats,
            out=out,
        )

    def _reduce(
        self,
        x: RegisterTensor,
        *,
        dim: Optional[int | Sequence[int]] = None,
        keepdim: bool,
        op: str,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        if dim is None:
            dim = range(len(x.shape))
        dims: list[int] = []
        if isinstance(dim, int):
            dims = [dim]
        else:
            dims = list(dim)
        dims = sorted(dims, reverse=True)
        cur = x
        for i, dim in enumerate(dims):
            if i < len(dims) - 1:
                cur = self._builder.reduce(cur, dim=dim, keepdim=keepdim, op=op)
            else:
                cur = self._builder.reduce(cur, dim=dim, keepdim=keepdim, op=op, out=out)
        return cur

    def sum(
        self,
        x: RegisterTensor,
        *,
        dim: Optional[int | Sequence[int]] = None,
        keepdim: bool = False,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Sum the elements along a specified dimension.

        This instruction computes the sum of the elements in the register tensor `x` along the specified dimension `dim`.
        If `keepdim` is set to `True`, the output tensor will have the same number of dimensions as the input tensor,
        with the specified dimension reduced to size 1. If `keepdim` is set to `False`, the output tensor will have
        the specified dimension removed, resulting in a tensor with one less dimension.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to reduce.
        dim: int | Sequence[int], optional
            The dimension(s) along which to compute the sum. They should be in range of [0, len(x.shape)). If not provided,
            all dimensions will be reduced.
        keepdim: bool, optional
            Whether to keep the reduced dimension in the output tensor. If `True`, the output tensor will have the
            same number of dimensions as the input tensor, with the specified dimension reduced to size 1. If `False`,
            the output tensor will have the specified dimension removed, resulting in a tensor with one less dimension.
            Default is `False`.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the sum of the elements along the specified dimension.
        """
        return self._reduce(x, dim=dim, keepdim=keepdim, op="sum", out=out)

    def max(
        self,
        x: RegisterTensor,
        *,
        dim: Optional[int | Sequence[int]] = None,
        keepdim: bool = False,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute the maximum value along a dimension.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to reduce.
        dim: int | Sequence[int], optional
            The dimension(s) along which to compute the maximum value. They should be in range of [0, len(x.shape)). If not provided,
            all dimensions will be reduced.
        keepdim: bool, optional
            Whether to keep the reduced dimension in the output tensor. If `True`, the output tensor will have the
            same number of dimensions as the input tensor, with the specified dimension reduced to size 1. If `False`,
            the output tensor will have the specified dimension removed, resulting in a tensor with one less dimension.
            Default is `False`.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the maximum value along the specified dimension.
        """
        return self._reduce(x, dim=dim, keepdim=keepdim, op="max", out=out)

    def min(
        self,
        x: RegisterTensor,
        *,
        dim: Optional[int | Sequence[int]] = None,
        keepdim: bool = False,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute the minimum value along a dimension.

        This instruction computes the minimum value of the elements in the register tensor `x` along the specified
        dimension `dim`. If `keepdim` is set to `True`, the output tensor will have the same number of dimensions as
        the input tensor, with the specified dimension reduced to size 1. If `keepdim` is set to `False`, the output
        tensor will have the specified dimension removed, resulting in a tensor with one less dimension.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to reduce.
        dim: int | Sequence[int], optional
            The dimension(s) along which to compute the minimum value. They should be in range of [0, len(x.shape)). If not provided,
            all dimensions will be reduced.
        keepdim: bool, optional
            Whether to keep the reduced dimension in the output tensor. If `True`, the output tensor will have the
            same number of dimensions as the input tensor, with the specified dimension reduced to size 1. If `False`,
            the output tensor will have the specified dimension removed, resulting in a tensor with one less dimension.
            Default is `False`.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the minimum value along the specified dimension.
        """
        return self._reduce(x, dim=dim, keepdim=keepdim, op="min", out=out)

    def any(
        self,
        x: RegisterTensor,
        *,
        dim: Optional[int | Sequence[int]] = None,
        keepdim: bool = False,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute whether there is a non-zero element along a dimension.

        This instruction computes whether any element in the register tensor `x` along the specified
        dimension `dim` is non-zero. If `keepdim` is set to `True`, the output tensor will have the same number of dimensions as the
        input tensor, with the specified dimension reduced to size 1. If `keepdim` is set to `False`, the output tensor will have the
        specified dimension removed, resulting in a tensor with one less dimension.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to reduce. It must be a boolean tensor.
        dim: int | Sequence[int], optional
            The dimension(s) along which to compute the any value. They should be in range of [0, len(x.shape)). If not provided,
            all dimensions will be reduced.
        keepdim: bool, optional
            Whether to keep the reduced dimension in the output tensor. If `True`, the output tensor will have the
            same number of dimensions as the input tensor, with the specified dimension reduced to size 1. If `False`,
            the output tensor will have the specified dimension removed, resulting in a tensor with one less dimension.
            Default is `False`.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the boolean result of whether any element in the register tensor `x` along the specified
            dimension `dim` is non-zero. The data type of the output tensor will be boolean.
        """
        if x.dtype != boolean:
            raise InstructionError("The input tensor must be a boolean tensor.")
        return self._reduce(x, dim=dim, keepdim=keepdim, op="any", out=out)

    def all(
        self,
        x: RegisterTensor,
        *,
        dim: Optional[int | Sequence[int]] = None,
        keepdim: bool = False,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute whether all elements are non-zero along a dimension.

        This instruction computes whether all elements in the register tensor `x` along the specified
        dimension `dim` are non-zero. If `keepdim` is set to `True`, the output tensor will have the same number of dimensions as the
        input tensor, with the specified dimension reduced to size 1. If `keepdim` is set to `False`, the output tensor will have the
        specified dimension removed, resulting in a tensor with one less dimension.

        Parameters
        ----------
        x: RegisterTensor
            The register tensor to reduce. It must be a boolean tensor.
        dim: int | Sequence[int], optional
            The dimension along which to compute the all value. This should be a valid dimension index for the tensor. If not provided,
            all dimensions will be reduced.
        keepdim: bool, optional
            Whether to keep the reduced dimension in the output tensor. If `True`, the output tensor will have the same number of dimensions as the
            input tensor, with the specified dimension reduced to size 1. If `keepdim` is set to `False`, the output tensor will have the
            specified dimension removed, resulting in a tensor with one less dimension.
            Default is `False`.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the boolean result of whether all elements in the register tensor `x` along the specified
            dimension `dim` are non-zero. The data type of the output tensor will be boolean.
        """
        if x.dtype != boolean:
            raise InstructionError("The input tensor must be a boolean tensor.")
        return self._reduce(x, dim=dim, keepdim=keepdim, op="all", out=out)

    def add(self, lhs: RegisterTensor, rhs: RegisterTensor, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        """Add two register tensors element-wise.

        This instruction computes the element-wise addition of two register tensors `lhs` and `rhs`. The result is a new
        register tensor with the same dtype.

        This instruction supports broadcasting, so if the shapes of `lhs` and `rhs` are not the same, they will be
        broadcasted to a common shape before performing the addition. The broadcasting rules are similar to those in
        NumPy and PyTorch, where dimensions of size 1 can be expanded to match the other tensor's shape.

        Parameters
        ----------
        lhs: RegisterTensor
            The left-hand side register tensor to add.
        rhs: RegisterTensor
            The right-hand side register tensor to add.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the element-wise sum of `lhs` and `rhs`. The shape of the output
            tensor will be determined by the broadcasting rules applied to `lhs` and `rhs`.
        """
        return self._builder.add(lhs, rhs, out=out)

    def maximum(self, lhs: RegisterTensor, rhs: RegisterTensor, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        """Compute the element-wise maximum.

        This instruction computes the element-wise maximum of two register tensors `lhs` and `rhs`. The result is a new
        register tensor with the same dtype.

        This instruction supports broadcasting, so if the shapes of `lhs` and `rhs` are not the same, they will be
        broadcasted to a common shape before performing the maximum operation. The broadcasting rules are similar to
        those in NumPy and PyTorch, where dimensions of size 1 can be expanded to match the other tensor's shape.

        Parameters
        ----------
        lhs: RegisterTensor
            The left-hand side register tensor to compare.
        rhs: RegisterTensor
            The right-hand side register tensor to compare.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the element-wise maximum of `lhs` and `rhs`. The shape of the output
            tensor will be determined by the broadcasting rules applied to `lhs` and `rhs`.
        """
        return self._builder.maximum(lhs, rhs, out=out)

    def where(
        self,
        condition: RegisterTensor,
        x: RegisterTensor | Expr | int | float,
        y: RegisterTensor | Expr | int | float,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Select elements from two tensors based on a condition.

        This instruction selects elements from two tensors `x` and `y` based on a boolean condition tensor. For each
        element in the condition tensor, if the condition is `True`, the corresponding element from `x` is selected,
        otherwise the corresponding element from `y` is selected. The result is a new register tensor with the same
        dtype as `x` and `y`.

        This instruction supports broadcasting, so if the shapes of `x` and `y` are not the same, they will be
        broadcasted to a common shape before performing the selection. The broadcasting rules are similar to those in
        NumPy and PyTorch, where dimensions of size 1 can be expanded to match the other tensor's shape.

        Parameters
        ----------
        condition: RegisterTensor
            The boolean register tensor that determines which elements to select from `x` and `y`. Its shape must match
            the shape of `x` and `y` after broadcasting.
        x: RegisterTensor | Expr | int | float
            The register tensor or expression to select elements from when the condition is `True`. If `x` is not a
            register tensor, it will be converted to a zero-dimensional register tensor with the same dtype as `y`.
        y: RegisterTensor | Expr | int | float
            The register tensor or expression to select elements from when the condition is `False`. If `y` is not a
            register tensor, it will be converted to a zero-dimensional register tensor with the same dtype as `x`.
        out: RegisterTensor, optional
            The register tensor to store the result. If not provided, a new register tensor will be allocated.

        Returns
        -------
        ret: RegisterTensor
            The register tensor containing the selected elements based on the condition. The shape of the output tensor
            will be determined by the broadcasting rules applied to `x` and `y`. The dtype of the output tensor will be
            the same as that of `x` and `y`.
        """
        if not isinstance(condition, RegisterTensor):
            cond_expr = as_expr(condition)
            condition = self._builder.allocate_register(dtype=boolean, shape=(), f_init=lambda _: cond_expr)
        if not isinstance(x, RegisterTensor):
            x_expr = as_expr(x)
            x = self._builder.allocate_register(dtype=infer_type(x), shape=(), f_init=lambda _: x_expr)
        if not isinstance(y, RegisterTensor):
            y_expr = as_expr(y)
            y = self._builder.allocate_register(dtype=infer_type(y), shape=(), f_init=lambda _: y_expr)
        if condition.dtype != boolean:
            raise InstructionError("Condition must be a boolean tensor, got {}".format(condition.dtype))
        if x.dtype != y.dtype:
            raise InstructionError("The types of x and y must match, got {} and {}".format(x.dtype, y.dtype))
        return self._builder.where(condition, x, y, out=out)

    def lock_semaphore(self, semaphore: Expr, value: Expr | int) -> None:
        """Lock semaphore with a specified value.

        This instruction locks the given semaphore with a specified value. It will block the thread until the semaphore
        is set to the specified value. The semaphore is a global int32 variable and `semaphore` should be an expression
        that evaluates to the address of the semaphore variable.

        Parameters
        ----------
        semaphore: Expr
            The expression that evaluates to the address of the semaphore variable.
        value: Expr | int
            The value to lock the semaphore with. This can be an integer or an expression that evaluates to an 32-bit
            signed integer.
        """
        self._builder.lock_semaphore(semaphore, value)

    def release_semaphore(self, semaphore: Expr, value: Expr | int) -> None:
        """Release semaphore with a specified value.

        This instruction releases the given semaphore with a specified value. It will set the semaphore to the
        specified value and make it visible to other thread blocks. The semaphore is a global int32 variable and
        `semaphore` should be an expression that evaluates to the address of the semaphore variable.

        Parameters
        ----------
        semaphore: Expr
            The expression that evaluates to the address of the semaphore variable.
        value: Expr | int
            The value to release the semaphore with. This can be an integer or an expression that evaluates to an
            32-bit signed integer.
        """
        self._builder.release_semaphore(semaphore, value)

    def cluster_sync(self) -> None:
        """Perform a cluster-wide synchronization.

        All threads in the entire block cluster will wait at this point until all threads reach this instruction.
        """
        self._builder.cluster_sync()

    def sync(self) -> None:
        """Perform a synchronization.

        The thread block will continue execution only after all previous instructions finished executing.
        """
        self._builder.syncthreads()

    @typing.overload
    def annotate_layout(self, tensor: RegisterTensor, layout: RegisterLayout) -> None:
        """Annotate the layout of a register tensor.

        This instruction annotates the layout of a register tensor with a specified layout. The `layout` parameter
        is an instance of `RegisterLayout` that defines how the tensor's data is organized among the threads in the
        thread block.

        This layout will be used to guide the layout inference process.

        Parameters
        ----------
        tensor: RegisterTensor
            The tensor to annotate.
        layout: RegisterLayout
            The layout to annotate the tensor with.
        """
        ...

    @typing.overload
    def annotate_layout(self, tensor: SharedTensor, layout: SharedLayout) -> None:  # type: ignore[overload-cannot-match]
        """Annotate the layout of a shared tensor.

        This instruction annotates the layout of a shared tensor with a specified layout. The `layout` parameter
        is an instance of `SharedLayout` that defines how the tensor's data is organized in shared memory.

        Parameters
        ----------
        tensor: SharedTensor
            The tensor to annotate.
        layout: SharedLayout
            The layout to annotate the tensor with.
        """
        pass

    def annotate_layout(self, tensor: RegisterTensor | SharedTensor, layout: RegisterLayout | SharedLayout) -> None:
        self._builder.annotate_layout(tensor, layout)

    def print_tensor(self, msg: str, tensor: Tensor, fmt: Optional[str] = None) -> None:
        """Print a tensor with a message.

        This instruction prints the contents of a tensor along with a message. The `msg` parameter is a string that
        will be printed before the tensor contents.

        The `fmt` parameter is an optional format string that specifies how the tensor elements should be formatted when
        printed.

        Parameters
        ----------
        msg: str
            The message to print before the tensor contents.
        tensor: Tensor
            The tensor to print. It can be any tensor type, including `RegisterTensor`, `GlobalTensor`, or `SharedTensor`.
        fmt: str
            The format string to use when printing the tensor elements. If not provided, a default format will be used.
            It should be a valid format specifier in C-style format used in `printf` function.
            The default format is determined according to the data type of the tensor elements.

            - int32: "%5d"
            - float16: "%5.2f"
            - float32: "%6.3f"
            - boolean: "%1d"
        """
        self._builder.print_tensor(msg=msg, tensor=tensor, fmt=fmt)

    def printf(self, fstring: str, *args: Expr | int | float) -> None:
        """Print a formatted string.

        This instruction prints a formatted string to the standard output. The `fstring` parameter is a format string
        that specifies how the output should be formatted. The `args` parameter is a variable-length argument list that
        contains the values to be formatted according to the `fstring`.

        Parameters
        ----------
        fstring: str
            The format string that specifies how the output should be formatted. It can contain format specifiers
            similar to those used in C-style `printf` function.
        args: Expr | int | float
            The values to be formatted according to the `fstring`. These can be expressions, integers, or floats.
            The number and types of `args` should match the format specifiers in `fstring`.

        See Also
        --------
        :c:func:`printf`: The C-style printf function for formatted output. For its documentation, refer to the
        `printf reference <https://cplusplus.com/reference/cstdio/printf/>`_.

        """
        self._builder.printf(fstring, *args)

    def assign(self, dst: RegisterTensor, src: RegisterTensor) -> None:
        """Assign the value of src tensor to dst tensor.

        This instruction copies the contents of the source register tensor `src` to the destination register tensor `dst`.

        Parameters
        ----------
        dst: RegisterTensor
            The destination tensor.
        src: RegisterTensor
            The source tensor.
        """
        if dst.dtype != src.dtype:
            raise InstructionError("The dtypes of dst and src must match, got {} and {}".format(dst.dtype, src.dtype))
        self._builder.assign_register(dst, src)
