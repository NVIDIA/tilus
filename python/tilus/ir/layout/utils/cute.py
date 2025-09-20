from __future__ import annotations
from hidet.utils import prod
from typing import Sequence, Any, Union
import functools
from dataclasses import dataclass
from hidet.ir.expr import Expr
from tilus.extensions.hidet.ir.utils.index_transform import index_deserialize



# class NestedTuple:
#     def __init__(self, items: Sequence[Int | NestedTuple]):
#         self.items: list[int | NestedTuple] = items
#         self._size: int | None = None
    
#     def __str__(self) -> str:
#         return '[' + ', '.join(str(item) for item in self.items) + ']'
    
#     def __len__(self) -> int:
#         return len(self.items)

#     def __mul__(self, other: NestedTuple) -> NestedTuple:
#         return NestedTuple([a * b for a, b in zip(self.items, other.items)])
    
#     def __getitem__(self, index: int) -> NestedTuple | Int:
#         return self.items[index]

#     def __call__(self, *coords: NestedTuple | Int | Sequence[Int]) -> NestedTuple:
#         if isinstance(coords, Sequence) and not isinstance(coords, NestedTuple):
#             coords = NestedTuple(coords)
#         return self.specialize(coords)

#     @staticmethod
#     def is_congruent(a: NestedTuple, b: NestedTuple) -> bool:
#         if len(a.items) != len(b.items):
#             return False
#         for a_item, b_item in zip(a.items, b.items):
#             if isinstance(a_item, NestedTuple) and isinstance(b_item, NestedTuple):
#                 if not NestedTuple.is_congruent(a_item, b_item):
#                     return False
#         return True

#     def specialize(self, *coords: NestedTuple | Int) -> NestedTuple:
#         while isinstance(coords, Sequence) and len(coords) == 1:
#             coords = coords[0]

#         if isinstance(coords, int):
#             sizes: list[Int] = []
#             for item in self.items:
#                 if isinstance(item, NestedTuple):
#                     sizes.append(item.size)
#                 else:
#                     sizes.append(item)
#             coords = index_deserialize(coords, sizes, ranks=list(reversed(range(len(sizes)))))

#         if len(self) != len(coords):
#             raise ValueError(f'can not specialize {coords} with {self}')

#         items = []
#         for i, item in enumerate(self.items):
#             if isinstance(item, NestedTuple):
#                 items.append(item.specialize(coords[i]))
#             else:
#                 items.append(coords[i])
#         return NestedTuple(items)


#     @functools.cache
#     def sum(self) -> int:
#         ret = 0
#         for item in self.items:
#             if isinstance(item, NestedTuple):
#                 ret += item.sum()
#             else:
#                 ret += item
#         return ret

#     @functools.cache
#     def product(self) -> int:
#         ret = 1
#         for item in self.items:
#             if isinstance(item, NestedTuple):
#                 ret *= item.product()
#             else:
#                 ret *= item

Int = Expr | int
IntTuple = Int | Sequence[Union[Int, 'IntTuple']]

def is_congruent(a: IntTuple, b: IntTuple) -> bool:
    if not isinstance(a, Sequence) and not isinstance(b, Sequence):
        return True
    if len(a) != len(b):
        return False
    return all(is_congruent(a_item, b_item) for a_item, b_item in zip(a, b))

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
        return f'{self.shape}:{self.strides}'
    
    def __call__(self, *coords: IntTuple) -> Int:
        coords = specialize(coords, self.shape)
        print(f'coords: {coords}, strides: {self.strides}')
        ret = tuple_sum(tuple_multiply(coords, self.strides))
        print(f'ret: {ret}')
        return ret


class CuteSwizzle:
    def __init__(self, bbits: int, mbase: int, sshift: int):
        self.bbits: int = bbits
        self.mbase: int = mbase
        self.sshift: int = sshift
    
    def __str__(self) -> str:
        return f'Swizzle<{self.bbits}, {self.mbase}, {self.sshift}>'

    def __call__(self, offset: Int) -> Int:
        if self.bbits == 0:
            return offset
        else:
            return offset ^ ((offset & (((1 << self.bbits) - 1) << (self.mbase + self.sshift))) >> self.sshift)
    

def cute_layout(shape: IntTuple, strides: IntTuple) -> CuteLayout:
    return CuteLayout(shape, strides)

if __name__ == "__main__":
    shape = (4, 5)
    strides = (1, 4)
    layout = cute_layout(shape, strides)
    for i in range(4):
        for j in range(5):
            print(specialize(i * 5 + j, layout.shape), end=' ')
            print(layout(i + j * 4), end=' ')
            print(layout(i, j), end=' ')
        print()
