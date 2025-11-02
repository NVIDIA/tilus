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
from typing import TypeVar, Generic, cast
from hidet.ir.builders.stmt_builder import StmtBuilder, StmtScope
from hidet.ir.expr import Expr, Var
from hidet.ir.stmt import ForStmtAttr

T = TypeVar("T")

class TypedStmtScope(StmtScope, Generic[T]):
    def __enter__(self) -> T:
        return cast(T, super().__enter__())

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)
    

class TypedStmtBuilder(StmtBuilder):
    def for_range(self, extent: Expr | int, *, attr: str | ForStmtAttr | None = None) -> TypedStmtScope[Var]:
        return cast(TypedStmtScope[Var], super().for_range(extent, attr=attr))
