# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from __future__ import annotations

import dataclasses
import string
from typing import Any, Callable, List, Optional, Tuple, Union

from tilus.hidet.ir.expr import Call, Expr, Var
from tilus.hidet.ir.node import Node
from tilus.hidet.ir.stmt import Stmt
from tilus.hidet.ir.type import BaseType


def check_func_name(name: str):
    if len(name) == 0:
        raise ValueError("Do not allow empty function name.")
    for c in name:
        if not (c in string.ascii_lowercase or c in string.ascii_uppercase or c in string.digits or c in "_"):
            raise ValueError("Cannot use {} in function name".format(repr(c)))


Dim3Like = Union[int, Expr, Tuple[Union[int, Expr], ...], List[Union[int, Expr]]]


@dataclasses.dataclass(frozen=True)
class FuncAttrs:
    """Typed attributes attached to a :class:`Function`.

    The launch-configuration fields (``grid_dim``, ``cluster_dim``, ``block_dim``,
    ``dynamic_smem_bytes``, ``min_blocks``) are only meaningful when the owning
    function's ``kind`` is ``"cuda_kernel"`` or ``"hip_kernel"``. The device is
    determined by ``Function.kind``; there is no separate per-device prefix.
    """

    grid_dim: Optional[Dim3Like] = None
    cluster_dim: Optional[Dim3Like] = None
    block_dim: Optional[Dim3Like] = None
    dynamic_smem_bytes: Union[int, Expr, None] = None
    min_blocks: Optional[int] = None

    def replace(self, **kwargs: Any) -> FuncAttrs:
        """Return a copy of this FuncAttrs with the given fields replaced."""
        return dataclasses.replace(self, **kwargs)

    def map(self, fn: Callable[[Any], Any]) -> FuncAttrs:
        """Return a new FuncAttrs with ``fn`` applied to every non-None field.

        Identity-preserving: if ``fn`` returns the same object for every field
        (``fn(value) is value``), ``self`` is returned unchanged. This is
        required by fixed-point rewriters that loop until the IR stops
        changing — producing a fresh FuncAttrs on every no-op pass would
        cause the loop to never converge.
        """
        changes = {}
        changed = False
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            new_value = fn(value)
            if new_value is not value:
                changes[f.name] = new_value
                changed = True
        return dataclasses.replace(self, **changes) if changed else self


class Function(Node):
    def __init__(
        self,
        name: str,
        params: List[Var],
        body: Stmt,
        ret_type: BaseType,
        kind: str,
        attrs: Optional[FuncAttrs] = None,
    ):
        check_func_name(name)
        assert isinstance(kind, str) and kind in [
            "cuda_kernel",
            "cuda_internal",
            "hip_kernel",
            "hip_internal",
            "cpu_kernel",
            "cpu_internal",
            "public",
        ]
        self.name: str = name
        self.kind: str = kind
        self.params: List[Var] = params
        self.body: Stmt = body
        self.ret_type: BaseType = ret_type
        self.attrs: FuncAttrs = attrs if attrs is not None else FuncAttrs()

    def __call__(self, *args, **kwargs) -> Call:
        raise ValueError("Can only call script function in another script function, or lower it to execute.")
