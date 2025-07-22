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
from . import register_layout_ops as ops
from .global_layout import GlobalLayout, global_column_major, global_compose, global_row_major, global_strides
from .register_layout import RegisterLayout, locate_at, register_layout, visualize_layout
from .register_layout_ops import (
    auto_local_spatial,
    column_local,
    column_spatial,
    compose,
    concat,
    divide,
    flatten,
    local,
    permute,
    reduce,
    reshape,
    spatial,
    squeeze,
    unsqueeze,
)
from .shared_layout import SharedLayout, shared_column_major, shared_compose, shared_row_major
from .utils import LayoutOperationError
