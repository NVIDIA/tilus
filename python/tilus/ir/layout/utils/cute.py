from __future__ import annotations

from typing import Sequence, Union

from hidet.ir.expr import Expr
from hidet.utils import prod
from tilus.extensions.hidet.ir.utils.index_transform import index_deserialize

Int = Expr | int
IntTuple = Int | Sequence[Union[Int, "IntTuple"]]


def is_congruent(a: IntTuple, b: IntTuple) -> bool:
    if not isinstance(a, Sequence) and not isinstance(b, Sequence):
        return True
    if len(a) != len(b):  # type: ignore[arg-type]
        return False
    return all(is_congruent(a_item, b_item) for a_item, b_item in zip(a, b))  # type: ignore[arg-type]


def tuple_multiply(a: IntTuple, b: IntTuple) -> IntTuple:
    if isinstance(a, Sequence) and isinstance(b, Sequence):
        return tuple(tuple_multiply(a_item, b_item) for a_item, b_item in zip(a, b))
    elif not isinstance(a, Sequence) and not isinstance(b, Sequence):
        return a * b
    else:
        raise ValueError(f"Invalid a or b: {a} or {b}")


def tuple_product(a: IntTuple) -> Int:
    return prod(tuple_product(item) for item in a) if isinstance(a, Sequence) else a


def tuple_sum(a: IntTuple) -> Int:
    return sum(tuple_sum(item) for item in a) if isinstance(a, Sequence) else a


def specialize(coords: IntTuple, shape: IntTuple) -> IntTuple:
    while isinstance(coords, Sequence) and len(coords) == 1:
        coords = coords[0]
    if isinstance(coords, Sequence) and isinstance(shape, Sequence):
        if len(coords) != len(shape):
            raise ValueError(f"Invalid coords or shape: {coords} or {shape}")
        return tuple(specialize(coord, shape) for coord, shape in zip(coords, shape))
    elif not isinstance(coords, Sequence) and not isinstance(shape, Sequence):
        return coords
    elif not isinstance(coords, Sequence) and isinstance(shape, Sequence):
        sizes = [tuple_product(item) for item in shape]
        return index_deserialize(coords, sizes, ranks=list(reversed(range(len(sizes)))))
    else:
        raise ValueError(f"Invalid coords or shape: {coords} or {shape}")


class CuteLayout:
    def __init__(self, shape: IntTuple, strides: IntTuple):
        self.shape: IntTuple = shape
        self.strides: IntTuple = strides

        if not is_congruent(shape, strides):
            raise ValueError("Shape and strides must be congruent")

    def __str__(self) -> str:
        return f"{self.shape}:{self.strides}"

    def __call__(self, *coords: IntTuple) -> Int:
        coords = specialize(coords, self.shape)
        ret = tuple_sum(tuple_multiply(coords, self.strides))
        return ret

    @property
    def flattened_shape(self) -> tuple[Int, ...]:
        if not isinstance(self.shape, Sequence):
            return (self.shape,)
        else:
            return tuple(tuple_product(item) for item in self.shape)


class CuteSwizzle:
    def __init__(self, bbits: int, mbase: int, sshift: int):
        self.bbits: int = bbits
        self.mbase: int = mbase
        self.sshift: int = sshift

    def __str__(self) -> str:
        return f"Swizzle<{self.bbits}, {self.mbase}, {self.sshift}>"

    def __call__(self, offset: Int) -> Int:
        if self.bbits == 0:
            return offset
        else:
            # 0xxxxYYYxxZZZxxxx
            # z_mask = ((1 << self.bbits) - 1) << self.mbase
            y_mask = ((1 << self.bbits) - 1) << (self.mbase + self.sshift)
            return offset ^ ((offset & y_mask) >> self.sshift)


class SwizzledCuteLayout:
    def __init__(self, layout: CuteLayout, swizzle: CuteSwizzle):
        self.layout: CuteLayout = layout
        self.swizzle: CuteSwizzle = swizzle

    def __str__(self) -> str:
        return str(self.swizzle) + " ○ " + str(self.layout)

    def __call__(self, *coords: IntTuple) -> Int:
        return self.swizzle(self.layout(*coords))


def cute_layout(shape: IntTuple, strides: IntTuple) -> CuteLayout:
    return CuteLayout(shape, strides)
