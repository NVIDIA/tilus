"""
This module contains utility functions that only depend on the Python standard library.
"""

from typing import List, Optional, Any, Callable, Sequence, Iterable
import itertools


def serial_imap(func: Callable, jobs: Sequence[Any], num_workers: Optional[int] = None) -> Iterable[Any]:
    yield from map(func, jobs)


def cdiv(a, b):
    return (a + (b - 1)) // b


def idiv(a: int, b: int):
    """
    Integer division with checking of proper division.
    """
    assert a % b == 0, "can not properly divide: {} // {}".format(a, b)
    return a // b


def floor_log2(n: int) -> int:
    ret = 0
    while n > 1:
        n //= 2
        ret += 1
    return ret


def select_bits(mask: int, left: int, right: int) -> int:
    # [left, right)
    return (mask >> left) & ((1 << (right - left)) - 1)


def factorize_decomposition(n: int) -> List[int]:
    assert n >= 1
    if n == 1:
        return []
    factors = []
    i = 2
    while i <= n:
        while n % i == 0:
            factors.append(i)
            n //= i
        i += 1
    return factors


def nbytes_from_nbits(nbits: int) -> int:
    assert nbits % 8 == 0
    return nbits // 8


def ranked_product(*iterables, ranks: Sequence[int]):
    assert set(ranks) == set(range(len(iterables)))
    reverse_ranks = {rank: i for i, rank in enumerate(ranks)}
    sorted_ranks_iterables = sorted(zip(ranks, iterables), key=lambda x: x[0])
    sorted_iterables = [iterable for _, iterable in sorted_ranks_iterables]
    for sorted_indices in itertools.product(*sorted_iterables):
        ranked_indices = [(reverse_ranks[i], sorted_indices[i]) for i in range(len(sorted_indices))]
        ranked_indices = sorted(ranked_indices, key=lambda x: x[0])
        indices = [index for _, index in ranked_indices]
        yield indices


def normalize_filename(filename: str) -> str:
    remap = {"/": "_", ".": "_", " ": "", "\t": "", "\n": "", "(": "", ")": "", ",": "_"}
    for k, v in remap.items():
        filename = filename.replace(k, v)
    # replace continuous _ with single _
    filename = filename.replace("__", "_")

    return filename
