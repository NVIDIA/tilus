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

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, TypeAlias, TypeVar

from hidet.ir.expr import Expr

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

    # the following functions should only be called in the __call__ function to construct the script program


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
