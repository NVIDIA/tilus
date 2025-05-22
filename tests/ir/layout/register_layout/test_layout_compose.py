import pytest
from tilus.ir.layout.register_layout_ops import compose, register_layout, repeat, spatial


@pytest.mark.parametrize(
    "outer, inner, expect",
    [
        [
            spatial(2, 4),
            repeat(2, 1),
            register_layout(shape=[4, 4], mode_shape=[2, 2, 4], spatial_modes=[0, 2], local_modes=[1]),
        ],
    ],
)
def test_compose(outer, inner, expect):
    actual = compose(outer, inner)
    assert actual == expect, f"Composition failed: {outer} / {inner}, expect {expect}, got {actual}"
