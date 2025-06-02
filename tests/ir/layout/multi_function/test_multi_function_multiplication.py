import pytest
from tilus.ir.layout.mfunction import MultiFunction, multi_function


@pytest.mark.parametrize(
    "a, b, expected",
    [
        [
            multi_function([2, 3, 4], [2, 3, 4], [0, 2]),
            multi_function([8], [2, 2, 2], [0, -5, 2]),
            multi_function([2, 3, 4], [2, 3, 2, 2], [0, -5, 3]),
        ],
        [
            multi_function([2, 3, 4], [2, 3, 4], [0, 2, -2]),
            multi_function([16], [2, 2, 2, 2], [0, -5, 2]),
            multi_function([2, 3, 4], [2, 3, 2, 2], [0, -5, 3]),
        ],
        [
            multi_function([2, 3, 4], [2, 3, 4], [0, 2, -2]),
            multi_function([16], [2, 2, 2, 2], [0, -5, 2, 3]),
            multi_function([2, 3, 4], [2, 3, 2, 2], [0, -5, 3, -2]),
        ],
    ],
)
def test_multi_function_multiplication(
    a: MultiFunction,
    b: MultiFunction,
    expected: MultiFunction,
):
    actual = a * b
    assert actual == expected, f"Expected {expected}, but got {actual} for {a} * {b}"
