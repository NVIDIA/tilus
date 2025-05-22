import pytest
from tilus.ir.layout.register_layout_ops import divide, repeat, spatial


@pytest.mark.parametrize(
    "lhs, rhs, expect",
    [
        [spatial(2, 4), spatial(1, 2), spatial(2, 2)],
        [spatial(2, 4).repeat(2, 1), repeat(2, 1), spatial(2, 4)],
        [spatial(2, 4).repeat(2, 1), spatial(1, 2).repeat(2, 1), spatial(2, 2)],
    ],
)
def test_divide(lhs, rhs, expect):
    actual = divide(lhs, rhs)
    assert actual == expect, f"Divide failed: {lhs} / {rhs}, expect {expect}, got {actual}"
