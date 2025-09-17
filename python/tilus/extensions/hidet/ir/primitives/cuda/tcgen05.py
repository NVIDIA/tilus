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
from typing import no_type_check

from hidet.ir.dtypes import uint32
from hidet.ir.expr import Expr
from hidet.ir.primitives.func import call_primitive_func
from hidet.ir.stmt import asm
from hidet.utils import initialize

from tilus.extensions.hidet.ir.primitives.utils import register_primitive_function_decorator


def resolve_tcgen05_relinquish_alloc_permit(cta_group: int) -> str:
    assert cta_group in (1, 2)
    return "cuda_tcgen05_relinquish_alloc_permit_cta_group_" + str(cta_group)


def resolve_tcgen05_alloc(cta_group: int) -> str:
    assert cta_group in (1, 2)
    return "cuda_tcgen05_alloc_cta_group_" + str(cta_group)


@initialize()
def register_tcgen05_instructions():
    from hidet.lang import attrs, script

    for cta_group in [1, 2]:

        @register_primitive_function_decorator
        @no_type_check
        @script
        def tcgen05_relinquish_alloc_permit_():
            attrs.func_name = resolve_tcgen05_relinquish_alloc_permit(cta_group)
            attrs.func_kind = "cuda_internal"
            asm("tcgen05.relinquish_alloc_permit.cta_group::{}.sync.aligned;".format(cta_group), is_volatile=True)

        @register_primitive_function_decorator
        @no_type_check
        @script
        def tcgen05_alloc_(dst: uint32, num_columns: uint32):
            attrs.func_name = resolve_tcgen05_alloc(cta_group)
            attrs.func_kind = "cuda_internal"
            asm(
                "tcgen05.alloc.cta_group::{}.sync.aligned.shared::cta.b32 [%0], %1;".format(cta_group),
                inputs=[dst, num_columns],
                is_volatile=True,
            )


def tcgen05_relinquish_alloc_permit(cta_group: int) -> Expr:
    func_name = resolve_tcgen05_relinquish_alloc_permit(cta_group)
    return call_primitive_func(func_name, [])


def tcgen05_alloc(dst: Expr, num_columns: Expr, cta_group: int) -> Expr:
    func_name = resolve_tcgen05_alloc(cta_group)
    return call_primitive_func(func_name, [dst, num_columns])
