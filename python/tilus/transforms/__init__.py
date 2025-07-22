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
from typing import Sequence

from .base import Pass, PassContext, apply_transforms
from .bound_aware_simplify import bound_aware_simplify_pass
from .declare_to_let import declare_to_let_pass
from .inject_print_instruction import inject_print_instruction_pass
from .layout_inference import layout_inference_pass
from .lower_load_store import lower_load_store_pass
from .lower_param_only_expr import lower_param_only_expr_pass
from .lower_to_load_matrix import lower_to_load_matrix_pass
from .scalar_analyze import analyze_scalar_pass


def get_default_passes() -> list[Pass]:
    return [
        declare_to_let_pass(),
        lower_param_only_expr_pass(),
        analyze_scalar_pass(),
        layout_inference_pass(),
        lower_to_load_matrix_pass(),
        lower_load_store_pass(),
        bound_aware_simplify_pass(),
        analyze_scalar_pass(),
    ]
