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
from typing import Any, Mapping, TypeVar, Union

from hidet.ir.node import Node
from hidet.ir.tools import rewrite as original_rewrite

_K = TypeVar("_K", bound=Node)
_V = TypeVar("_V", bound=Node)


def rewrite(
    node: Union[Node, tuple, list, dict], rewrite_map: Mapping[_K, _V], clone_internal_var: bool = False
) -> Any:
    return original_rewrite(node, rewrite_map, clone_internal_var)  # type: ignore
