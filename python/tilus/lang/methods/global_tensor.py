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
from hidet.ir.expr import Var
from tilus.ir.builders import StmtBuilder
from tilus.ir.tensor import GlobalTensor


class GlobalTensorWithMethods(GlobalTensor):
    def __init__(self, tensor: GlobalTensor, builder: StmtBuilder):
        super().__init__(tensor.dtype, tensor.layout)
        self.tensor = tensor
        self.builder = builder

    def item_ptr(self) -> Var:
        return self.builder.tensor_item_ptr(self.tensor, space="global")

    def item(self) -> Var:
        return self.builder.tensor_item_value(self.tensor)
