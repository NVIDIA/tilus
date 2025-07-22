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
import numpy as np
from hidet.ir.expr import Var
from tilus.ir.utils.veceval import vectorized_evaluate


def test_vectorized_evaluate():
    from hidet.ir.dtypes import int32

    # Example usage
    x = Var("x", int32)
    y = Var("y", int32)
    expr = x * y + 2

    var2value = {x: np.array([1, 2, 3]), y: np.array([4, 5, 6])}

    result = vectorized_evaluate(expr, var2value)
    assert np.all(result == np.array([6, 12, 20]))
