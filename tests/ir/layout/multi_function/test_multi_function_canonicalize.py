import pytest
from tilus.ir.layout.mfunction import canonicalize, multi_function


@pytest.mark.parametrize(
    "a, expected",
    [
        (
            multi_function([2, 3], [2, 1, 3], [0, 2]),
            multi_function([2, 3], [2, 3], [0, 1]),
        ),
        (
            multi_function([24, 5, 6], [2, 3, 4, 5, 6], [0, 1, 2, 4]),
            multi_function([24, 5, 6], [24, 5, 6], [0, 2]),
        ),
    ],
)
def test_multi_function_canonicalization(a, expected):
    actual = canonicalize(a)
    assert actual == expected, f"Expected {expected}, but got {actual} for {a}"
