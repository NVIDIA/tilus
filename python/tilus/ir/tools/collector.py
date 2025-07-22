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
from typing import Any, Sequence

from tilus.ir.functors import IRVisitor
from tilus.ir.node import IRNode


def collect(node: IRNode, types: Sequence[Any]) -> list[Any]:
    visitor = IRVisitor()
    visitor.visit(node)

    types = tuple(types)
    ret = []

    for node in visitor.memo.keys():
        if isinstance(node, types):
            ret.append(node)
    return ret
