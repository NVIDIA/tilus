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
from typing import List, Optional

from hidet.ir.expr import Dereference, Expr, Var, cast, var
from hidet.ir.type import BaseType


def deref(v: Expr, derefed_type: Optional[BaseType] = None) -> Expr:
    if derefed_type is not None:
        v = cast(v, ~derefed_type)
    return Dereference(v)


def index_vars(num_vars: int) -> List[Var]:
    """Create a list of index variables with given number of variables.

    Parameters
    ----------
    num_vars: int
        The number of index variables to create.

    Returns
    -------
    ret: List[Var]
        The list of created index variables.
    """
    default_names = ["i", "j", "k", "p", "q", "r", "s", "t", "u", "v"]
    if num_vars < len(default_names):
        return [var(default_names[i]) for i in range(num_vars)]
    else:
        return [var("i") for _ in range(num_vars)]


def reinterpret(value: Expr, target_type: BaseType) -> Expr:
    return cast(~value, ~target_type)[0]
