import pytest
import tilus

from tilus.ir.layout.shared_layout import SharedLayout, canonicalize_shared_layout

@pytest.mark.parametrize(
    "layout, expected",
    [
        (
            SharedLayout.create(
                shape=[32, 16],
                mode_shape=[4, 8, 2, 2, 4],
                mode_strides=[64, 8, 256, 4, 1],
                optional_swizzle=None
            ),
            SharedLayout.create(
                shape=[32, 16],
                mode_shape=[32, 2, 8],
                mode_strides=[8, 256, 1],
                optional_swizzle=None
            ),
        ),
    ],
)
def test_canonicalize_shared_layout(layout, expected):
    canonicalized = canonicalize_shared_layout(layout)
    assert canonicalized == expected


if __name__ == "__main__":
    pytest.main([__file__])
