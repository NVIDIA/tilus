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
from typing import Callable, Sequence, TypeVar
from hidet.ir.expr import Expr
from hidet.ir.tools import simplify_to_int

ArgType = TypeVar("ArgType")
ReturnType = TypeVar("ReturnType")


def group_function_argument(f: Callable[..., ReturnType]) -> Callable[[Sequence[ArgType]], ReturnType]:
    def wrapped(args: Sequence[ArgType]) -> ReturnType:
        return f(*args)

    return wrapped

def normalize_blocks_per_cluster(blocks_per_cluster) -> tuple[int, int, int]:
    if isinstance(blocks_per_cluster, (Expr, int)):
        blocks_per_cluster = simplify_to_int(blocks_per_cluster)
        return blocks_per_cluster, 1, 1
    elif isinstance(blocks_per_cluster, (list, tuple)):
        blocks_per_cluster = [simplify_to_int(b) for b in blocks_per_cluster]
        while len(blocks_per_cluster) < 3:
            blocks_per_cluster.append(1)
        if len(blocks_per_cluster) > 3:
            raise ValueError("The length of blocks_per_cluster should not be greater than 3.")
        return tuple(blocks_per_cluster)  # type: ignore
    else:
        raise ValueError("The type of blocks_per_cluster should be int, Expr, list or tuple.")

