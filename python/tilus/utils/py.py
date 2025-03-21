"""
This module contains utility functions that only depend on the Python standard library.
"""

import itertools
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence


def serial_imap(func: Callable, jobs: Sequence[Any], num_workers: Optional[int] = None) -> Iterable[Any]:
    yield from map(func, jobs)


def cdiv(a, b):
    return (a + (b - 1)) // b


def idiv(a: int, b: int) -> int:
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


def ranked_product(*iterables: Any, ranks: Sequence[int]) -> Iterator[List[Any]]:
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


def to_snake_case(name: str) -> str:
    """
    Convert a PascalCase string (e.g., 'NameLikeClass') to snake_case (e.g., 'name_like_class').

    Parameters
    ----------
    name: str
        The input string in PascalCase.

    Returns
    -------
    ret: str
        The converted string in snake_case.
    """
    result: list[str] = []
    for i, char in enumerate(name):
        # If it's an uppercase letter and not the first character
        if char.isupper() and i > 0:
            # Add an underscore before it if the previous char wasn't already an underscore
            if result[-1] != "_":
                result.append("_")
        result.append(char.lower())

    return "".join(result)


def relative_to_with_walk_up(source: Path, target: Path) -> Path:
    """
    Compute the relative path from source to target, allowing walking up the directory tree.
    Similar to Path.relative_to(..., walk_up=True) in Python 3.12+.

    Parameters
    ----------
    source: Path
        The starting path (Path object).
    target: Path
        The target path (Path object).

    Returns
    -------
    ret: Path
        A relative Path object from source to target.
    """
    source = source.resolve()
    target = target.resolve()

    # Convert paths to their absolute components
    source_parts = list(source.parts)
    target_parts = list(target.parts)

    # Find the common prefix length
    common_len = 0
    for s, t in zip(source_parts, target_parts):
        if s != t:
            break
        common_len += 1

    # Number of steps to walk up from source to the common ancestor
    walk_up_count = len(source_parts) - common_len

    # Relative path components: walk up with ".." and then append remaining target parts
    relative_parts = [".."] * walk_up_count + target_parts[common_len:]

    if not relative_parts:
        return Path(".")

    return Path(*relative_parts)
