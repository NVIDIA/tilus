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
"""The module contains utility functions that only depend on the Python standard library."""

import itertools
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence


def serial_imap(func: Callable, jobs: Sequence[Any], num_workers: Optional[int] = None) -> Iterable[Any]:
    yield from map(func, jobs)


def cdiv(a, b):
    return (a + (b - 1)) // b


def idiv(a: int, b: int) -> int:
    """Integer division with checking of proper division."""
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


def unique_file_name(pattern: str) -> Optional[str]:
    """Given a pattern like './results/exp/report_%d.txt' and returns a unique file name like `./results/exp/report_1.txt`."""
    import os

    if pattern.count("%d") == 0:
        os.makedirs(os.path.dirname(pattern), exist_ok=True)
        return pattern
    else:
        assert pattern.count("%d") == 1
        os.makedirs(os.path.dirname(pattern), exist_ok=True)

        i = 0
        while True:
            file_name = pattern % i
            if not os.path.exists(file_name):
                return file_name
            i += 1


# ---------------------------------------------------------------------------
# Functions below are ported from hidet/utils/py.py to be self-contained
# (no hidet or tilus.hidet imports).
# ---------------------------------------------------------------------------


def prod(seq: Iterable) -> Any:
    """Compute the product of all elements in *seq* (returns 1 for empty)."""
    seq = list(seq)
    if len(seq) == 0:
        return 1
    else:
        c = seq[0]
        for i in range(1, len(seq)):
            c = c * seq[i]
        return c


def gcd(a: int, b: int, *args: int) -> int:
    """
    Get the greatest common divisor of non-negative integers.

    Parameters
    ----------
    a: int
        The lhs operand.
    b: int
        The rhs operand.

    Returns
    -------
    ret: int
        The greatest common divisor.
    """
    if len(args) > 0:
        return gcd(gcd(a, b), *args)
    assert a >= 0 and b >= 0
    return a if b == 0 else gcd(b, a % b)


def lcm(a: int, b: int) -> int:
    """
    Get the least common multiple of non-negative integers a and b.

    Parameters
    ----------
    a: int
        The lhs operand.
    b: int
        The rhs operand.

    Returns
    -------
    ret: int
        The least common multiple.
    """
    return a // gcd(a, b) * b


def is_power_of_two(n: int) -> bool:
    """
    Check if an integer is a power of two: 1, 2, 4, 8, 16, 32, ...

    Parameters
    ----------
    n: int
        The integer to check.

    Returns
    -------
    ret: bool
        True if n is a power of two, False otherwise.
    """
    return n > 0 and (n & (n - 1)) == 0


def _is_immutable(obj):
    if isinstance(obj, (int, float, str, tuple)):
        return True
    # Lazy imports to avoid circular dependencies; gracefully return False
    # when the hidet IR layer is not available.
    try:
        from tilus.hidet.ir.expr import Constant

        if isinstance(obj, Constant) and obj.type.is_tensor():
            return False
        if isinstance(obj, Constant):
            return True
    except Exception:
        pass
    return False


def same_list(lhs, rhs, use_equal=False):
    """Check whether two lists are element-wise identical (by ``is``) or equal."""
    if len(lhs) != len(rhs):
        return False
    for l, r in zip(lhs, rhs):
        if use_equal or _is_immutable(l) and _is_immutable(r):
            if l != r:
                return False
        else:
            if l is not r:
                return False
    return True


def initialize(*args, **kwargs):
    """Decorate an initialization function.

    After decorating with this function, the initialization function will be called after the definition.

    Parameters
    ----------
    args:
        The positional arguments of initializing.
    kwargs:
        The keyword arguments of initializing.

    Returns
    -------
    ret:
        A decorator that will call given function with args and kwargs,
        and return None (to prevent this function to be called again).
    """

    def decorator(f):
        f(*args, **kwargs)

    return decorator


def str_indent(msg: str, indent: int = 0) -> str:
    """Indent every line of *msg* by *indent* spaces."""
    lines = msg.split("\n")
    lines = [" " * indent + line for line in lines]
    return "\n".join(lines)


def repeat_until_converge(func, obj, limit=None):
    """Repeatedly apply *func* to *obj* until the result stops changing (identity check)."""
    i = 0
    while True:
        i += 1
        orig_obj = obj
        obj = func(obj)
        if obj is orig_obj:
            return obj
        if limit is not None and i >= limit:
            return obj


class COLORS:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    MAGENTA = "\033[95m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def green(v, fmt="{}"):
    return COLORS.OKGREEN + fmt.format(v) + COLORS.ENDC


def cyan(v, fmt="{}"):
    return COLORS.OKCYAN + fmt.format(v) + COLORS.ENDC


def blue(v, fmt="{}"):
    return COLORS.OKBLUE + fmt.format(v) + COLORS.ENDC


def red(v, fmt="{}"):
    return COLORS.FAIL + fmt.format(v) + COLORS.ENDC


def bold(v, fmt="{}"):
    return COLORS.BOLD + fmt.format(v) + COLORS.ENDC


def nocolor(s: str) -> str:
    """Strip ANSI color codes from a string."""
    for value in COLORS.__dict__.values():
        if isinstance(value, str) and value[0] == "\033":
            s = s.replace(value, "")
    return s
