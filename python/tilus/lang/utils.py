from typing import Callable, Sequence, TypeVar

ArgType = TypeVar("ArgType")
ReturnType = TypeVar("ReturnType")


def group_function_argument(f: Callable[..., ReturnType]) -> Callable[[Sequence[ArgType]], ReturnType]:
    def wrapped(args: Sequence[ArgType]) -> ReturnType:
        return f(*args)

    return wrapped
