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
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
tilus.hidet.utils — re-exports utilities needed by the copied hidet code.

Most functions are forwarded from tilus.utils.py (the canonical, self-contained
copies).  Sub-modules (doc, namer, structure, ...) live alongside this __init__.
"""

from . import doc, git_utils, namer, py, stack_limit, structure
from .py import (
    COLORS,
    blue,
    bold,
    cdiv,
    cyan,
    gcd,
    green,
    initialize,
    is_power_of_two,
    lcm,
    nocolor,
    prod,
    red,
    repeat_until_converge,
    same_list,
    str_indent,
)
from .structure import DirectedGraph
