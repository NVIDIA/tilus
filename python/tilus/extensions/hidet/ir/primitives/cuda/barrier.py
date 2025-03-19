from typing import List, Optional, Union

from typing_extensions import no_type_check

from hidet.ir.dtypes import int32
from hidet.ir.expr import Expr
from hidet.ir.func import Function
from hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from hidet.utils import initialize


@initialize()
def register_functions():
    from hidet.lang import asm, attrs, script
    from hidet.lang.types import int32

    for aligned in [False, True]:
        for mode in ["arrive", "sync", "sync_all"]:
            func_name = "barrier_{}{}".format(mode, "_aligned" if aligned else "")

            if mode == "sync_all":
                template = "barrier.sync{} %0;".format(".aligned" if aligned else "")

                @no_type_check
                @script
                def barrier_func(barrier: int32):
                    attrs.func_kind = "cuda_internal"
                    attrs.func_name = func_name

                    asm(template, inputs=[barrier], is_volatile=True)

            else:
                template = "barrier.sync{} %0, %1;".format(".aligned" if aligned else "")

                @no_type_check
                @script
                def barrier_func(barrier: int32, count: int32):
                    attrs.func_kind = "cuda_internal"
                    attrs.func_name = func_name

                    asm(template, inputs=[barrier, count], is_volatile=True)

            assert isinstance(barrier_func, Function)
            register_primitive_function(name=barrier_func.name, func_or_type=barrier_func)


def _barrier(barrier: Union[int, Expr], count: Optional[Union[int, Expr]], aligned: bool, mode: str) -> Expr:
    # resolve function name
    func_name = "barrier_{}{}".format(mode, "_aligned" if aligned else "")

    # call the function
    args: List[Expr] = [int32.constant(barrier)]
    if count is not None:
        args.append(int32.constant(count))
    return call_primitive_func(func_name, args=args)


def barrier_sync(barrier: Union[int, Expr], count: Optional[Union[int, Expr]] = None, aligned: bool = False) -> Expr:
    """
    Performs barrier synchronization and communication within a CTA.

    The threads will synchronize at the named barrier.

    See Also
    --------
    The PTX ISA documentation for the `barrier` instruction:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier

    Parameters
    ----------
    barrier:
        The named barrier to synchronize on. This must be an integer from 0 to 15.

    count: Optional[int]
        The number of threads to synchronize. If not provided, all threads in the CTA will synchronize.

    aligned:
        When specified, it indicates that all threads in CTA will execute the same barrier instruction.
    """
    mode = "sync_all" if count is None else "sync"
    return _barrier(barrier, count, aligned, mode=mode)


def barrier_arrive(barrier: Union[int, Expr], count: Union[int, Expr], aligned: bool = False) -> Expr:
    """
    Performs barrier synchronization and communication within a CTA.

    The threads will mark their arrival at the named barrier but will not be blocked.

    See Also
    --------
    The PTX ISA documentation for the `barrier` instruction:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier

    Parameters
    ----------
    barrier: Union[int, Expr]
        The named barrier to synchronize on. This must be an integer from 0 to 15.

    count: Union[int, Expr]
        The number of threads to synchronize.

    aligned: bool
        When specified, it indicates that all threads in CTA will execute the same barrier instruction.
    """
    return _barrier(barrier, count, aligned, mode="arrive")
