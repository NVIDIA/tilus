import pytest
from tilus.ir.layout.register_layout_ops import divide, local, spatial


@pytest.mark.parametrize(
    "lhs, rhs, expect",
    [
        [spatial(2, 4), spatial(1, 2), spatial(2, 2)],
        [spatial(2, 4).local(2, 1), local(2, 1), spatial(2, 4)],
        [spatial(2, 4).local(2, 1), spatial(1, 2).local(2, 1), spatial(2, 2)],
    ],
)
def test_divide(lhs, rhs, expect):
    actual = divide(lhs, rhs)
    assert actual == expect, f"Divide failed: {lhs} / {rhs}, expect {expect}, got {actual}"
