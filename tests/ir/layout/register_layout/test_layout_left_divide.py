# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from tilus.ir.layout.register_layout_ops import left_divide, local, spatial


@pytest.mark.parametrize(
    "layout, lhs_divisor, expect",
    [
        [spatial(2, 4), spatial(2, 1), spatial(1, 4)],
        [local(2, 1).spatial(2, 4), local(2, 1), spatial(2, 4)],
        [spatial(2, 2).local(2, 1), spatial(2, 2), local(2, 1)],
        [spatial(2, 4), spatial(2, 2), spatial(1, 2)],
    ],
)
def test_left_divide(layout, lhs_divisor, expect):
    actual = left_divide(layout, lhs_divisor)
    assert actual == expect, f"Left divide failed: {layout} \\ {lhs_divisor}, expect {expect}, got {actual}"


