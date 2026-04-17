# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Callable, Iterable, Literal, Optional, Sequence, Union

from tilus.hidet.ir.dtypes import boolean
from tilus.hidet.ir.expr import Constant, Expr, Var, as_expr
from tilus.hidet.ir.primitives.cuda.vars import blockIdx, gridDim
from tilus.hidet.ir.tools import infer_type
from tilus.hidet.ir.type import DataType
from tilus.ir.inst import InstructionError
from tilus.ir.layout import GlobalLayout, RegisterLayout, SharedLayout
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor, Tensor
from tilus.lang.constructs.contexts import ThreadGroupContext
from tilus.lang.constructs.structs import Dim3

from .base import InstructionGroup


class RootInstructionGroup(InstructionGroup):
    @property
    def blockIdx(self) -> Dim3:
        """Get the block index of the current thread block."""
        return Dim3(blockIdx.x, blockIdx.y, blockIdx.z)  # type: ignore[attr-defined]

    @property
    def gridDim(self) -> Dim3:
        """Get the grid dimension of the kernel."""
        return Dim3(gridDim.x, gridDim.y, gridDim.z)  # type: ignore[attr-defined]

    @property
    def current_thread_begin(self) -> int:
        """Get the beginning thread index of the current thread group."""
        if len(self._builder.tg_stack.thread_begin) == 0:
            raise RuntimeError("No thread group context found.")
        return self._builder.tg_stack.thread_begin[-1]

    @property
    def current_thread_end(self) -> int:
        """Get the ending thread index of the current thread group."""
        if len(self._builder.tg_stack.thread_end) == 0:
            raise RuntimeError("No thread group context found.")
        return self._builder.tg_stack.thread_end[-1]

    @property
    def current_num_threads(self) -> int:
        """Get the number of threads in the current thread group."""
        if len(self._builder.tg_stack.thread_begin) == 0 or len(self._builder.tg_stack.thread_end) == 0:
            raise RuntimeError("No thread group context found.")
        return self._builder.tg_stack.thread_end[-1] - self._builder.tg_stack.thread_begin[-1]

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

    def range(
        self,
        start: Expr | int,
        end: Optional[Expr | int] = None,
        step: Optional[Expr | int] = None,
        /,
        *,
        unroll: Optional[Literal["all"] | int] = None,
    ) -> Iterable[Var]:
        """Create an iterator used in a for loop.

        This function creates an iterator that can be used in a for loop. It is similar to the built-in `range` function,
        but provides additional control like unrolling the loop.


        Parameters
        ----------
        start: Expr | int
            The starting value of the iterator.
        end: Expr | int, optional
            The end value of the iterator. If not provided, it is assumed to be 0 and `start` is used as the end value.
        step: Expr | int, optional
            The step value of the iterator. If not provided, it defaults to 1.
        unroll: Literal["all"] | int, optional
            The unrolling factor for the loop. If set to "all", the loop will be fully unrolled. If set to an integer,
            the loop will be unrolled by that factor. If not provided, no unrolling hint will be applied.

        Returns
        -------
        ret: Iterable[Var]
            The iterator that can be used in a for loop. It yields `Var` objects representing the loop indices.

        Examples
        --------
        We can use this function to create a for loop iterator, similar to the built-in `range` function:

        .. code-block:: python

            # the following two loops are equivalent
            for i in range(10):
                ...
            for i in self.range(10):
                ...

            # we can also specify the start, end, and step values
            for i in range(1, 10, 2):
                ...
            for i in self.range(1, 10, 2):
                ...

            # we can also specify the unrolling factor
            # unroll the loop completely
            for i in self.range(1, 10, 2, unroll="all"):
                ...

            # or unroll the loop by a factor of 4
            for i in self.range(1, 10, 2, unroll=4):
                ...

        """
        from tilus.lang.constructs.loops import range

        # the cast is to make the type checker happy
        return typing.cast(Iterable[Var], range(start, end, step, unroll=unroll))

    def thread_group(self, thread_begin: int, num_threads: int) -> ThreadGroupContext:
        """Create a thread group context.

        This method creates a thread group context that is used to narrow down the threads that execute the instructions
        within the context.

        Syntax:

        .. code-block:: python

            class MyScript(tilus.Script):

                def __call__(self, ...):
                    # instructions executed by all threads in the thread block
                    ...
                    with self.thread_group(thread_begin, num_threads=num_threads):
                        # instructions executed by threads in the specified thread group starting from `thread_begin`
                        # and including `num_threads` threads
                        ...
                        with self.thread_group(...):
                            # we can continue to partition the current thread group into sub thread groups
                            ...
                        ...
                        self.sync()  # synchronize all threads in the current thread group
                        ...

                    # instructions executed by all threads in the thread block
                    ...

        At the root level of the kernel, there is one thread group that includes all threads in the thread block.
        We can partition the threads in the current thread group into multiple sub thread groups by specifying the
        first thread using `thread_begin` and the number of threads in each sub thread group using the `num_threads`
        parameter.

        All instructions within the context will be executed by all threads in the specified thread group.

        Parameters
        ----------
        thread_begin: int
            The index of the first thread in the thread group.
        num_threads: int
            The number of threads in the thread group.

        Returns
        -------
        ret: ThreadGroupContext
            The thread group context created.
        """
        return ThreadGroupContext(self._builder, thread_begin=thread_begin, num_threads=num_threads)

    def single_thread(self, thread: int = -1) -> ThreadGroupContext:
        """Create a thread group context with only one thread.

        By default (``thread=-1``), uses elect-any semantics: the hardware is
        free to pick any single thread in the current group, enabling the
        back-end to emit ``elect.sync`` / uniform-predicate code.

        Pass an explicit thread index (e.g., ``thread=0``) to pin execution to
        a specific thread.

        Parameters
        ----------
        thread : int
            Thread index within the current group, or ``-1`` for elect-any.

        Returns
        -------
        ret: ThreadGroupContext
            The thread group context created.
        """
        if thread != -1 and thread >= self.current_num_threads:
            raise InstructionError(
                "The thread index must be less than the number of threads in the current thread group"
            )
        return self.thread_group(thread_begin=thread, num_threads=1)

    def single_warp(self, warp: int = 0) -> ThreadGroupContext:
        """Create a thread group context with a single warp (32 threads).

        This method is equivalent to `thread_group(<first-thread-in-a-warp>, num_threads=32)` that creates a thread group
        context with 32 threads. All instructions within the context will be executed by only one warp.

        Returns
        -------
        ret: ThreadGroupContext
            The thread group context created.
        """
        if warp * 32 >= self.current_num_threads:
            raise InstructionError(
                "The warp index must be such that the first thread in the warp is less than the number of threads in the current thread group"
            )
        return self.thread_group(thread_begin=warp * 32, num_threads=32)

    def warp_group(self, warp_begin: int, num_warps: int) -> ThreadGroupContext:
        """Create a thread group context with multiple warps.

        This method is equivalent to `thread_group(<first-thread-in-the-first-warp>, num_threads=num_warps*32)` that creates a thread group
        context with multiple warps. All instructions within the context will be executed by the specified number of warps.

        Returns
        -------
        ret: ThreadGroupContext
            The thread group context created.
        """
        if warp_begin * 32 >= self.current_num_threads:
            raise InstructionError(
                "The warp_begin index must be such that the first thread in the first warp is less than the number of threads in the current thread group"
            )
        if (warp_begin + num_warps) * 32 > self.current_num_threads:
            raise InstructionError(
                "The number of warps must be such that the last thread in the last warp is less than the number of threads in the current thread group"
            )
        return self.thread_group(thread_begin=warp_begin * 32, num_threads=num_warps * 32)

    @staticmethod
    def static_assert(cond: bool | Expr, msg: str) -> None:
        """Assert a compile-time condition.

        Raises ``AssertionError`` at compile time if the condition is false. The condition
        must be a compile-time constant (a ``bool`` or a ``Constant`` expression).

        Parameters
        ----------
        cond: bool | Expr
            The compile-time condition to check. Must be a constant value.
        msg: str
            The error message to display if the assertion fails.
        """
        if not isinstance(cond, Constant) and not isinstance(cond, bool):
            raise ValueError("Static assert condition must be a constant")
        if not cond:
            raise AssertionError(msg)

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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        self._builder.free_shared(tensor)

    def reshape_shared(self, tensor: SharedTensor, shape: Sequence[int]) -> SharedTensor:
        """Reshape a shared tensor.

        Parameters
        ----------
        tensor: SharedTensor
            The shared tensor to reshape.
        shape: Sequence[int]
            The new shape of the shared tensor.

        Returns
        -------
        ret: SharedTensor
            The reshaped shared tensor.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        return self._builder.reshape_shared(tensor=tensor, shape=shape)

    def copy_async(
        self,
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
        evict: Optional[str] = None,
        check_bounds: bool = True,
    ) -> None:
        """Asynchronously copy a tile from global memory to shared memory.

        Issues an ``cp.async`` transfer from a region of ``src`` (global) to ``dst`` (shared).
        Use ``copy_async_commit_group()`` and ``copy_async_wait_group()`` to synchronize.

        Out-of-bounds accesses are zero-filled by default. Set ``check_bounds=False`` to skip
        bounds checking when you can guarantee all accesses are in-bounds.

        Parameters
        ----------
        src: GlobalTensor
            The global tensor to copy from.
        dst: SharedTensor
            The shared tensor to copy to.
        offsets: Sequence[Expr | int]
            Starting offsets for each dimension of the global tensor. Length must match the
            rank of the global tensor.
        dims: Sequence[int], optional
            Which dimensions of the global tensor are being sliced. If not provided, defaults
            to all dimensions in order.
        evict: str, optional
            Cache eviction policy. Candidates:

            - ``'evict_normal'`` (default): normal eviction priority.
            - ``'evict_first'``: evict this data first; suitable for streaming access patterns.

        check_bounds: bool, optional
            If ``True`` (default), out-of-bounds accesses are zero-filled. If ``False``,
            bounds checking is skipped (caller must guarantee in-bounds access).

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 8.0+ (sm_80).
        - **PTX**: ``cp.async``
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 8.0+ (sm_80).
        """
        self._builder.copy_async_wait_all()

    def copy_async_commit_group(self):
        """Commit async copies into a group.

        This instruction commits all the pending asynchronous copy operations into a group.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 8.0+ (sm_80).
        - **PTX**: ``cp.async.commit_group``
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 8.0+ (sm_80).
        - **PTX**: ``cp.async.wait_group``
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 7.0+ (sm_70) for tensor core MMA, or any GPU for SIMT fallback.
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        return self._builder.unsqueeze(x, dim=dim, out=out)

    def flatten(self):
        """Flatten a register tensor into a 1-D tensor.

        .. note:: This instruction is not yet implemented.
        """
        self._builder

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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        return self._builder.transpose(x)

    def abs(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute the element-wise absolute value.

        Parameters
        ----------
        x: RegisterTensor
            Input tensor.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            Tensor with the same shape and dtype as ``x``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        return self._builder.abs(x, out=out)

    def exp(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute the element-wise natural exponential (e^x).

        Parameters
        ----------
        x: RegisterTensor
            Input tensor.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            Tensor with the same shape and dtype as ``x``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        return self._builder.exp(x, out=out)

    def exp2(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute the element-wise base-2 exponential (2^x).

        Parameters
        ----------
        x: RegisterTensor
            Input tensor.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            Tensor with the same shape and dtype as ``x``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        return self._builder.exp2(x, out=out)

    def log(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute the element-wise natural logarithm (ln x).

        Parameters
        ----------
        x: RegisterTensor
            Input tensor.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            Tensor with the same shape and dtype as ``x``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        return self._builder.log(x, out=out)

    def round(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Round each element to the nearest integer (round-to-nearest-even).

        Uses banker's rounding: if the fractional part is exactly 0.5, rounds to the nearest
        even integer.

        Parameters
        ----------
        x: RegisterTensor
            Input tensor.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            Tensor with the same shape and dtype as ``x``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        return self._builder.round(x, out=out)

    def square(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute the element-wise square (x^2).

        Parameters
        ----------
        x: RegisterTensor
            Input tensor.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            Tensor with the same shape and dtype as ``x``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        return self._builder.square(x, out=out)

    def sqrt(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute the element-wise square root.

        Parameters
        ----------
        x: RegisterTensor
            Input tensor.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            Tensor with the same shape and dtype as ``x``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        return self._builder.sqrt(x, out=out)

    def rsqrt(
        self,
        x: RegisterTensor,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Compute the element-wise reciprocal square root (1/sqrt(x)).

        Parameters
        ----------
        x: RegisterTensor
            Input tensor.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            Tensor with the same shape and dtype as ``x``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        return self._builder.rsqrt(x, out=out)

    def clip(
        self,
        x: RegisterTensor,
        min: Expr | int | float,
        max: Expr | int | float,
        *,
        out: Optional[RegisterTensor] = None,
    ) -> RegisterTensor:
        """Clip element values to the range [min, max].

        Parameters
        ----------
        x: RegisterTensor
            Input tensor.
        min: Expr | int | float
            Lower bound. Values below this are set to ``min``.
        max: Expr | int | float
            Upper bound. Values above this are set to ``max``.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            Tensor with the same shape and dtype as ``x``.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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
        """Sum elements along the specified dimension(s).

        Parameters
        ----------
        x: RegisterTensor
            Input tensor.
        dim: int | Sequence[int], optional
            Dimension(s) to reduce. If not provided, reduces all dimensions.
        keepdim: bool
            If ``True``, retains the reduced dimension with size 1. Default is ``False``.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            The reduced tensor.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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
        """Compute the maximum along the specified dimension(s).

        Parameters
        ----------
        x: RegisterTensor
            Input tensor.
        dim: int | Sequence[int], optional
            Dimension(s) to reduce. If not provided, reduces all dimensions.
        keepdim: bool
            If ``True``, retains the reduced dimension with size 1. Default is ``False``.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            The reduced tensor.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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
        """Compute the minimum along the specified dimension(s).

        Parameters
        ----------
        x: RegisterTensor
            Input tensor.
        dim: int | Sequence[int], optional
            Dimension(s) to reduce. If not provided, reduces all dimensions.
        keepdim: bool
            If ``True``, retains the reduced dimension with size 1. Default is ``False``.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            The reduced tensor.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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
        """Test whether any element is non-zero along the specified dimension(s).

        Parameters
        ----------
        x: RegisterTensor
            Input boolean tensor.
        dim: int | Sequence[int], optional
            Dimension(s) to reduce. If not provided, reduces all dimensions.
        keepdim: bool
            If ``True``, retains the reduced dimension with size 1. Default is ``False``.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            Boolean tensor with the reduction result.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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
        """Test whether all elements are non-zero along the specified dimension(s).

        Parameters
        ----------
        x: RegisterTensor
            Input boolean tensor.
        dim: int | Sequence[int], optional
            Dimension(s) to reduce. If not provided, reduces all dimensions.
        keepdim: bool
            If ``True``, retains the reduced dimension with size 1. Default is ``False``.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            Boolean tensor with the reduction result.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        if x.dtype != boolean:
            raise InstructionError("The input tensor must be a boolean tensor.")
        return self._reduce(x, dim=dim, keepdim=keepdim, op="all", out=out)

    def add(self, lhs: RegisterTensor, rhs: RegisterTensor, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        """Element-wise addition with broadcasting.

        Parameters
        ----------
        lhs: RegisterTensor
            Left operand.
        rhs: RegisterTensor
            Right operand.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            ``lhs + rhs``, broadcast to a common shape.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        return self._builder.add(lhs, rhs, out=out)

    def maximum(self, lhs: RegisterTensor, rhs: RegisterTensor, out: Optional[RegisterTensor] = None) -> RegisterTensor:
        """Element-wise maximum with broadcasting.

        Parameters
        ----------
        lhs: RegisterTensor
            Left operand.
        rhs: RegisterTensor
            Right operand.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            ``max(lhs, rhs)`` element-wise, broadcast to a common shape.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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
        """Select elements from ``x`` or ``y`` based on a boolean condition.

        Returns ``x`` where ``condition`` is ``True``, ``y`` otherwise. Supports broadcasting.
        Scalar values for ``x`` or ``y`` are automatically promoted to tensors.

        Parameters
        ----------
        condition: RegisterTensor
            Boolean tensor determining element selection.
        x: RegisterTensor | Expr | int | float
            Values selected where ``condition`` is ``True``.
        y: RegisterTensor | Expr | int | float
            Values selected where ``condition`` is ``False``.
        out: RegisterTensor, optional
            Output tensor. If not provided, a new tensor is allocated.

        Returns
        -------
        ret: RegisterTensor
            Tensor with the same dtype as ``x`` and ``y``, broadcast to a common shape.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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

        Notes
        -----
        - **Thread group**: Must be executed by a single thread (use ``self.single_thread()``).
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

        Notes
        -----
        - **Thread group**: Must be executed by a single thread (use ``self.single_thread()``).
        """
        self._builder.release_semaphore(semaphore, value)

    def sync(self) -> None:
        """Perform a synchronization.

        The thread block will continue execution only after all previous instructions finished executing.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **PTX**: ``bar.sync``
        """
        self._builder.syncthreads()

    @typing.overload
    def annotate_layout(self, tensor: RegisterTensor, layout: RegisterLayout) -> None: ...

    @typing.overload
    def annotate_layout(self, tensor: SharedTensor, layout: SharedLayout) -> None:  # type: ignore[overload-cannot-match]
        pass

    def annotate_layout(self, tensor: RegisterTensor | SharedTensor, layout: RegisterLayout | SharedLayout) -> None:
        """Annotate the layout of a register or shared tensor.

        This instruction annotates the layout of a register or shared tensor with a specified layout. The `layout` parameter
        is an instance of `RegisterLayout` or `SharedLayout` that defines how the tensor's data is organized among the threads in the
        thread block.

        This layout will be used to guide the layout inference process.

        Parameters
        ----------
        tensor: RegisterTensor | SharedTensor
            The tensor to annotate.
        layout: RegisterLayout | SharedLayout
            The layout to annotate the tensor with. The type of layout must match the type of tensor.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        self._builder.print_tensor(msg=msg, tensor=tensor, fmt=fmt)

    def printf(self, fstring: str, *args: Expr | int | float | str) -> None:
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


        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
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

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        if dst.dtype != src.dtype:
            raise InstructionError("The dtypes of dst and src must match, got {} and {}".format(dst.dtype, src.dtype))
        self._builder.assign_register(dst, src)

    def fast_divmod(self, a: Union[Expr, int], b: Union[Expr, int]) -> tuple:
        """Fast integer division and modulo using precomputed magic multiplier.

        Computes ``(a // b, a % b)`` for ``a >= 0`` and ``b > 0`` using the
        FastDivmod algorithm. The divisor ``b`` must be a grid-constant expression
        (known at kernel launch time). A Hidet IR pass will lower this to
        ``__umulhi(a, magic) >> shift`` with host-side precomputation of
        ``(magic, shift)`` from ``b``.

        Parameters
        ----------
        a : Expr
            The dividend (int32, must be >= 0).
        b : Expr
            The divisor (int32, must be > 0 and grid-constant).

        Returns
        -------
        quotient : Expr
            floor(a / b)
        remainder : Expr
            a % b (computed as a - quotient * b)

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        """
        from tilus.hidet.ir.primitives.cuda.fast_divmod import fastdiv

        if isinstance(b, Constant):
            q = a // b
            r = a % b
            return q, r
        else:
            q = fastdiv(a, b)
            r = a - q * b
            return q, r
