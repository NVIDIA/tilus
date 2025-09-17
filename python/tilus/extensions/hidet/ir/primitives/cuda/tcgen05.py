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

from hidet.ir.expr import Expr
from hidet.ir.func import Function
from hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from hidet.ir.stmt import asm
from hidet.utils import initialize


def resolve_tcgen05_relinquish_alloc_permit(cta_group: int) -> str:
    assert cta_group in (1, 2)
    return 'cuda_tcgen05_relinquish_alloc_permit_cta_group_' + str(cta_group)


@initialize()
def register_tcgen05_instructions():
    from hidet.lang import script, attrs

    for cta_group in [1, 2]:
        func_name = resolve_tcgen05_relinquish_alloc_permit(cta_group)
        template = "tcgen05.relinquish_alloc_permit.cta_group::{}.sync.aligned;".format(cta_group)
        @script
        def cuda_tcgen05_relinquish_alloc_permit():
            attrs.func_name = func_name
            attrs.func_kind = 'cuda_internal'
            asm(template, is_volatile=True, memory_fence=True)

        assert isinstance(cuda_tcgen05_relinquish_alloc_permit, Function)
        register_primitive_function(name=func_name, func_or_type=cuda_tcgen05_relinquish_alloc_permit)



def tcgen05_relinquish_alloc_permit(cta_group: int) -> Expr:
    func_name = resolve_tcgen05_relinquish_alloc_permit(cta_group)
    return call_primitive_func(func_name, [])

