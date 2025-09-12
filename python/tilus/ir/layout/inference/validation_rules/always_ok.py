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
from tilus.ir.instructions import (
    AllocateRegisterInst,
    AllocateSharedInst,
    CopyAsyncGenericInst,
    CopyAsyncInst,
    FormatPrintInst,
    FreeSharedInst,
    Instruction,
    LoadGlobalGenericInst,
    LoadGlobalInst,
    LoadSharedGenericInst,
    LoadSharedInst,
    PrintTensorInst,
    SharedSliceInst,
    StoreGlobalGenericInst,
    StoreGlobalInst,
    StoreSharedInst,
)
from tilus.ir.instructions.cuda.cp_async_bulk import CopyAsyncBulkGlobalToSharedInst, CopyAsyncBulkSharedToGlobalInst, CopyAsyncBulkGlobalToClusterSharedInst, CopyAsyncBulkSharedToClusterSharedInst
from tilus.ir.layout.inference.rule import LayoutValidationRule, register_rule


@register_rule(CopyAsyncBulkSharedToClusterSharedInst)
@register_rule(CopyAsyncBulkGlobalToClusterSharedInst)
@register_rule(CopyAsyncBulkGlobalToSharedInst)
@register_rule(CopyAsyncBulkSharedToGlobalInst)
@register_rule(CopyAsyncGenericInst)
@register_rule(CopyAsyncInst)
@register_rule(StoreGlobalGenericInst)
@register_rule(StoreSharedInst)
@register_rule(PrintTensorInst)
@register_rule(FormatPrintInst)
@register_rule(LoadGlobalInst)
@register_rule(LoadGlobalGenericInst)
@register_rule(StoreGlobalInst)
@register_rule(SharedSliceInst)
@register_rule(LoadSharedInst)
@register_rule(LoadSharedGenericInst)
@register_rule(FreeSharedInst)
@register_rule(AllocateRegisterInst)
@register_rule(AllocateSharedInst)
class AlwaysOkayRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: Instruction) -> bool:
        return True


# todo: the following instructions should have dedicated validation rules
class TemporaryAlwaysOkRule(LayoutValidationRule):
    @staticmethod
    def validate(inst: Instruction) -> bool:
        return True
