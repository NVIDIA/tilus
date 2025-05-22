import pytest
from tilus.ir.layout.register_layout_ops import register_layout, reshape, spatial


@pytest.mark.parametrize(
    "original, shape, expect",
    [
        [spatial(2, 4), [2, 2, 2], spatial(2, 2, 2)],
        [
            spatial(2, 4).local(3, 2).spatial(6, 8),
            [2, 3, 6, 4, 2, 8],
            register_layout(
                shape=[2, 3, 6, 4, 2, 8], mode_shape=[2, 3, 6, 4, 2, 8], spatial_modes=[0, 3, 2, 5], local_modes=[1, 4]
            ),
        ],
        [
            register_layout(
                shape=[2, 3, 6, 4, 2, 8], mode_shape=[2, 3, 6, 4, 2, 8], spatial_modes=[0, 3, 2, 5], local_modes=[1, 4]
            ),
            [36, 64],
            register_layout(
                shape=[36, 64], mode_shape=[2, 3, 6, 4, 2, 8], spatial_modes=[0, 3, 2, 5], local_modes=[1, 4]
            ),
        ],
    ],
)
def test_reshape(original, shape, expect):
    actual = reshape(original, shape)
    assert actual == expect, f"Reshape failed: {original} -> {shape}, expect {expect}, got {actual}"
