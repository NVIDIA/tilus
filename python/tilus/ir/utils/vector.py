from __future__ import annotations

from typing import Any, Callable, Generic, Iterable, Sequence, TypeVar, overload

T = TypeVar("T")


class Vector(Generic[T]):
    def __init__(self, items: Sequence[T]):
        self.items: Sequence[Any] = items

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self):
        raise ValueError("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")

    def __iter__(self):
        return iter(self.items)

    def __add__(self, other: Any) -> Vector[T]:
        return self.binary(self, other, lambda a, b: a + b)

    def __radd__(self, other: Any) -> Vector[T]:
        return self.binary(other, self, lambda a, b: a + b)

    def __sub__(self, other: Any) -> Vector[T]:
        return self.binary(self, other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> Vector[T]:
        return self.binary(other, self, lambda a, b: a - b)

    def __mul__(self, other: Any) -> Vector[T]:
        return self.binary(self, other, lambda a, b: a * b)

    def __truediv__(self, other: Any) -> Vector[T]:
        return self.binary(self, other, lambda a, b: a / b)

    def __rtruediv__(self, other: Any) -> Vector[T]:
        return self.binary(other, self, lambda a, b: a / b)

    def __floordiv__(self, other: Any) -> Vector[T]:
        return self.binary(self, other, lambda a, b: a // b)

    def __rfloordiv__(self, other: Any) -> Vector[T]:
        return self.binary(other, self, lambda a, b: a // b)

    def __mod__(self, other: Any) -> Vector[T]:
        return self.binary(self, other, lambda a, b: a % b)

    def __rmod__(self, other: Any) -> Vector[T]:
        return self.binary(other, self, lambda a, b: a % b)

    def __lt__(self, other: Any) -> Vector[bool]:
        return self.binary(self, other, lambda a, b: a < b)

    def __gt__(self, other: Any) -> Vector[bool]:
        return self.binary(self, other, lambda a, b: a > b)

    def __ge__(self, other: Any) -> Vector[bool]:
        return self.binary(self, other, lambda a, b: a >= b)

    def __le__(self, other: Any) -> Vector[bool]:
        return self.binary(self, other, lambda a, b: a <= b)

    def __eq__(self, other: Any) -> Vector[bool]:
        return self.binary(self, other, lambda a, b: a == b)

    def __ne__(self, other: Any) -> Vector[bool]:
        return self.binary(self, other, lambda a, b: a != b)

    def __and__(self, other: Any) -> Vector[bool]:
        return self.binary(self, other, lambda a, b: a and b)

    def __or__(self, other: Any) -> Vector[bool]:
        return self.binary(self, other, lambda a, b: a or b)

    @staticmethod
    def binary(lhs: Any, rhs: Any, op: Callable[[Any, Any], Any]) -> Vector[Any]:
        if not isinstance(lhs, (Vector, Sequence)):
            lhs = [lhs]
        if not isinstance(rhs, (Vector, Sequence)):
            rhs = [rhs]
        if len(lhs) != len(rhs) and len(lhs) != 1 and len(rhs) != 1:
            raise ValueError("Expect both vectors have the same length, or one of them is a scalar")
        if len(lhs) == 1 and len(rhs) > 1:
            lhs = [lhs[0]] * len(rhs)
        elif len(rhs) == 1 and len(lhs) > 1:
            rhs = [rhs[0]] * len(lhs)

        return Vector([op(a, b) for a, b in zip(lhs, rhs)])

    def all(self) -> bool:
        return all(self.items)

    def any(self) -> bool:
        return any(self.items)


@overload
def vector(seq: Sequence[T]) -> Vector[T]: ...


@overload
def vector(iterable: Iterable[T]) -> Vector[T]: ...


@overload
def vector(*items: T) -> Vector[T]: ...


def vector(*args):
    if len(args) == 1 and isinstance(args[0], Sequence):
        return Vector(args[0])
    elif len(args) == 1 and isinstance(args[0], Iterable):
        return Vector(list(args[0]))
    else:
        return Vector(args)
