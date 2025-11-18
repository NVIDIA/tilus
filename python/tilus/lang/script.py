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

import typing
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Optional, Sequence, TypeAlias, TypeVar

from hidet.ir.expr import Constant, Expr, Var
from hidet.ir.primitives.cuda.vars import blockIdx, gridDim

from tilus.lang.constructs.contexts import ThreadGroupContext
from tilus.lang.constructs.structs import Dim3
from tilus.lang.instructions import InstructionInterface
from tilus.lang.modules.cuda import cuda

if TYPE_CHECKING:
    from tilus.lang.instantiated_script import InstantiatedScript, JitInstance

Int: TypeAlias = int | Expr


class Attributes:
    def __init__(self):
        self.blocks: Optional[Sequence[Int] | Int] = None
        self.cluster_blocks: Sequence[Int] | Int = (1, 1, 1)
        self.warps: Optional[int] = None


class Script(InstructionInterface):
    """A script is a user-defined kernel function that can be compiled and executed on the GPU."""

    # the compiled program will print the instruction output of the specified block
    debug_block: Optional[tuple[int, int, int]] = None

    # specify the schedule used for debugging. it will override any autotune space
    debug_schedule: Optional[dict[str, Any]] = None

    def __new__(cls, *args, **kwargs) -> InstantiatedScript:  # type: ignore[no-untyped-def]
        from tilus.lang.instantiated_script import InstantiatedScriptCache

        instantiated_script: InstantiatedScript = InstantiatedScriptCache.get(
            script_cls=cls,
            script_args=args,
            script_kwargs=kwargs,
        )

        return instantiated_script

    def __init__(self) -> None:
        super().__init__()

        # attributes
        self._attrs: Attributes = Attributes()

        # modules
        self.cuda = cuda

    def __call__(self, *args, **kwargs):
        raise RuntimeError("This method should never be called.")

    def jit_instance_for(self, *args: object, **kwargs: object) -> JitInstance:
        """
        Instantiate the script program with the specified arguments and keyword arguments.

        Parameters
        ----------
        args:
            The positional arguments to the __call__ method.
        kwargs:
            The keyword arguments to the __call__ method.

        Returns
        -------
        ret: JitInstance
            The JIT instance for the script with given arguments.
        """
        raise RuntimeError("This method should never be called. See InstantiatedScript.jit_instance instead.")

    # the following properties should only be access in the __call__ function
    @property
    def attrs(self) -> Attributes:
        """Kernel attributes like number of blocks and warps.

        See :py:class:`Attributes <tilus.lang.Attributes>` for more details.
        """
        return self._attrs

    @property
    def blockIdx(self) -> Dim3:
        """Get the block index of the current thread block."""
        return Dim3(blockIdx.x, blockIdx.y, blockIdx.z)

    @property
    def gridDim(self) -> Dim3:
        """Get the grid dimension of the kernel."""
        return Dim3(gridDim.x, gridDim.y, gridDim.z)

    # the following functions should only be called in the __call__ function to construct the script program

    @staticmethod
    def range(
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

    @staticmethod
    def thread_group(group_index: int, group_size: int) -> ThreadGroupContext:
        """Create a thread group context.

        This method creates a thread group context that is used to narrow down the threads that execute the instructions
        within the context.

        Syntax:

        .. code-block:: python

            class MyScript(tilus.Script):

                def __call__(self, ...):
                    # instructions executed by all threads in the thread block
                    ...
                    with self.thread_group(group_index, group_size=group_size):
                        # instructions executed by threads in the specified thread group
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
        number of threads in each sub thread group using the `group_size` parameter.

        All instructions within the context will be executed by all threads in the specified thread group.

        Parameters
        ----------
        group_index: int
            The index of the thread group to be created. It must be in the range [0, num_groups).
        group_size: int
            The number of threads in each thread group.

        Returns
        -------
        ret: ThreadGroupContext
            The thread group context created.
        """
        return ThreadGroupContext(group_index=group_index, group_size=group_size)

    @staticmethod
    def single_thread() -> ThreadGroupContext:
        """Create a thread group context with only one thread.

        This method is equivalent `thread_group(<any-thread>, group_size=1)` that creates a thread group
        context with only one thread. All instructions within the context will be executed by only one thread.

        Returns
        -------
        ret: ThreadGroupContext
            The thread group context created.
        """
        return Script.thread_group(group_index=0, group_size=1)

    @staticmethod
    def static_assert(cond: bool | Expr, msg: str) -> None:
        if not isinstance(cond, Constant) and not isinstance(cond, bool):
            raise ValueError("Static assert condition must be a constant")
        if not cond:
            raise AssertionError(msg)


T = TypeVar("T")


def autotune(arg_names: str, arg_values: Sequence[Any]) -> Callable[[T], T]:
    """Annotate an autotune subspace for a tilus script.

    Parameters
    ----------
    arg_names: str
        The names of the arguments for autotuning, separated by commas.
    arg_values: Sequence[Any]
        The sequence of the choices for the autotune parameters. Each choice can be a single value or a sequence of
        values that match the names in `arg_names`.

    Returns
    -------
    ret: Callable[[Type[Script]], Type[Script]]
        The decorator that can be applied to a tilus script class for the marking of autotune parameters.
    """

    def decorator(script_cls: T) -> T:
        if not hasattr(script_cls, "_autotune_space"):
            setattr(script_cls, "_autotune_space", {})
        space = getattr(script_cls, "_autotune_space")
        names = [name.strip() for name in arg_names.split(",")]

        # check names and arg_values
        # 1. can not define the same name more than once
        if any(name in space for name in names):
            common_names = set(names) & set(space.keys())
            raise RuntimeError("Duplicated specification for parameters: {}".format(common_names))
        # 2. the arg_values should match the names during unpacking
        if not isinstance(arg_values, Sequence):
            raise TypeError("The arg_values values must be a sequence")
        for arg_value in arg_values:
            if len(names) > 1:
                if not isinstance(arg_value, Sequence) or len(arg_value) != len(names):
                    raise TypeError(
                        "Can not unpack the arg_values for arg_names\n"
                        f"  arg_names: {arg_names}\n"
                        f"  arg_value: {arg_value}"
                    )

        space[arg_names] = arg_values
        setattr(script_cls, "_autotune_space", space)

        # return functools.wraps(wrapped=script_cls, assigned=arg_names)(script_cls)
        return script_cls

    return decorator
