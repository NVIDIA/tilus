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
from typing import Type

from tilus.ir.layout.inference.rule import LayoutInferenceRule
from tilus.utils import initialize

from .inference_rules.allocate_shared import AllocateSharedRule
from .inference_rules.assign import AssignRule
from .inference_rules.cp_async import CopyAsyncRule
from .inference_rules.elementwise_binary import BinaryRule
from .inference_rules.elementwise_unary import UnaryRule
from .inference_rules.empty_rule import EmptyRule
from .inference_rules.ldst_global import LoadGlobalRule, StoreGlobalRule
from .inference_rules.load_shared import (
    LoadSharedInferRegisterRule,
    LoadSharedInferRowMajorSharedRule,
    LoadSharedInferSwizzledSharedRule,
)
from .inference_rules.mma_dot import MmaDotRule
from .inference_rules.reduce import ReduceRule
from .inference_rules.shared_slice import SharedSliceRule
from .inference_rules.store_shared import StoreSharedSwizzleRule
from .inference_rules.tcgen05.ldst import Tcgen05LoadRule, Tcgen05StoreRule
from .inference_rules.tcgen05.mma import Tcgen05MmaRule
from .inference_rules.tcgen05.copy import Tcgen05CopyRule
from .inference_rules.transform import SqueezeRule, UnsqueezeRule
from .inference_rules.transpose import TransposeRule
from .inference_rules.where import WhereRule

inference_order: list[list[Type[LayoutInferenceRule]]] = [
    [Tcgen05MmaRule],
    [MmaDotRule],
    [Tcgen05LoadRule, Tcgen05StoreRule],
    [Tcgen05CopyRule],
    [BinaryRule, UnaryRule],
    [LoadGlobalRule],
    [ReduceRule],
    [TransposeRule, SqueezeRule, UnsqueezeRule],
    [WhereRule],
    [AssignRule],
    [StoreGlobalRule],
    [EmptyRule],
    # shared memory rules
    [LoadSharedInferSwizzledSharedRule, StoreSharedSwizzleRule],
    [SharedSliceRule],
    [CopyAsyncRule],
    [LoadSharedInferRegisterRule],
    [LoadSharedInferRowMajorSharedRule],
    [AllocateSharedRule],
]

rule2order: dict[Type[LayoutInferenceRule], int] = {}


@initialize()
def init_rule_sort_key() -> None:
    count = 0
    for rule_group in inference_order:
        for rule in rule_group:
            rule2order[rule] = count
            count += 1
