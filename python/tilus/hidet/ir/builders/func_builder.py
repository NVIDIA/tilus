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
from typing import List, Optional, Sequence

from tilus.hidet.ir.expr import Var
from tilus.hidet.ir.func import FuncAttrs, Function
from tilus.hidet.ir.stmt import Stmt
from tilus.hidet.ir.type import VoidType

from .stmt_builder import StmtBuilder


class FunctionBuilder(StmtBuilder):
    def __init__(
        self,
        name: str,
        kind: str,
        ret_type=VoidType(),
        grid_dim=None,
        cluster_dim=None,
        block_dim=None,
        dynamic_smem_bytes=None,
        min_blocks=None,
        attrs: Optional[FuncAttrs] = None,
    ):
        super().__init__()
        self.name = name
        self.kind = kind
        self.params: List[Var] = []
        self.ret_type = ret_type
        self.func: Optional[Function] = None
        self.body: Optional[Stmt] = None

        base = attrs if attrs is not None else FuncAttrs()
        self.attrs: FuncAttrs = base.replace(
            **{
                key: value
                for key, value in (
                    ("grid_dim", grid_dim),
                    ("cluster_dim", cluster_dim),
                    ("block_dim", block_dim),
                    ("dynamic_smem_bytes", dynamic_smem_bytes),
                    ("min_blocks", min_blocks),
                )
                if value is not None
            }
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finish_func()

    def extend_params(self, params: Sequence[Var]):
        self.params.extend(params)

    def update_attrs(self, **kwargs) -> None:
        """Replace selected FuncAttrs fields on this builder."""
        self.attrs = self.attrs.replace(**kwargs)

    def set_body(self, body: Stmt):
        self.body = body

    def finish_func(self):
        assert self.func is None
        if self.body is None:
            self.body = self.finish()
        self.func = Function(
            self.name, kind=self.kind, params=self.params, body=self.body, ret_type=self.ret_type, attrs=self.attrs
        )

    def get(self) -> Function:
        assert self.func.body is not None
        return self.func
