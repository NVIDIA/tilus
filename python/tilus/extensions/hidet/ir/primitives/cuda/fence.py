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
from typing import Union, no_type_check

from hidet.ir.dtypes import boolean
from hidet.ir.expr import Expr
from hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from hidet.ir.stmt import asm
from hidet.lang import attrs, script, u32
from hidet.utils import initialize


@initialize()
def register_mbarrier_primitives():
    @no_type_check
    @script
    def cuda_fence_mbarrier_init_cluster():
        attrs.func_kind = "cuda_internal"
        asm("fence.mbarrier_init.release.cluster;", is_volatile=True, memory_fence=True)
    
    for func in [cuda_fence_mbarrier_init_cluster]:
        register_primitive_function(func.name, func)


def fence_mbarrier_init_cluster() -> Expr:
    """Issue a fence to initialize mbarrier in cluster scope.

    Returns
    -------
    Expr
        An expression representing the fence operation.
    """
    return call_primitive_func('cuda_fence_mbarrier_init_cluster', args=[])
