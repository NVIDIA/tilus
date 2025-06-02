import pytest
from tilus.ir.layout.mfunction import multi_function


@pytest.mark.parametrize(
    "fa, fb, expected",
    [
        (
            multi_function([2, 3, 4], [2, 3, 4], [0, 2, -3]),
            multi_function([2, 3, 4], [2, 3, 4], [0, 2, 1]),
            True,
        ),
        (
            multi_function([2, 3, 4], [2, 3, 4], [0, 2, 1]),
            multi_function([2, 3, 4], [2, 3, 4], [0, 2, -3]),
            False,
        ),
        (
            multi_function([2, 3], [2, 3], [0, -3]),
            multi_function([2, 3], [2, 3], [0, 1]),
            True,
        ),
    ],
)
def test_multi_function_cover(fa, fb, expected):
    actual = fa.cover(fb)
    assert actual == expected, f"Expected {expected} for cover({fa}, {fb}), but got {actual}"
