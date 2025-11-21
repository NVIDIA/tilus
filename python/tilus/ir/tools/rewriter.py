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
from typing import Any, Mapping

from tilus.ir.functors import IRRewriter
from tilus.ir.node import IRNode


class SimpleRewriter(IRRewriter):
    """A simple rewriter that applies a function to each instruction in the program."""

    def __init__(self, rewrite_map: Mapping[IRNode, Any]):
        super().__init__()
        self.memo.update(rewrite_map)


def rewrite(node: IRNode, rewrite_map: Mapping[Any, Any]) -> Any:
    """
    Rewrite the components of the given node using the provided rewrite map.

    Parameters
    ----------
    node: IRNode
        The node to rewrite.

    rewrite_map: Mapping[IRNode, IRNode]
        A mapping from nodes to their rewritten versions.

    Returns
    -------
    ret: Any
        The rewritten node.
    """
    rewriter = SimpleRewriter(rewrite_map)
    return rewriter.visit(node)
