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
from hidet.ir.type import BaseType, PointerType, TensorPointerType, TensorType


def is_addressable(tp_or_var):
    from hidet.ir.expr import Var

    if isinstance(tp_or_var, Var):
        tp = tp_or_var.type
    else:
        tp = tp_or_var
    return isinstance(tp, (PointerType, TensorPointerType, TensorType))


def get_base_type(tp: BaseType) -> BaseType:
    if isinstance(tp, PointerType):
        return tp.base_type
    elif isinstance(tp, TensorPointerType):
        return tp.tensor_type.dtype
    elif isinstance(tp, TensorType):
        return tp.dtype
    else:
        assert False
