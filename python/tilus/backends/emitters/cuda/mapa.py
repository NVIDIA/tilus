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
from tilus.backends.emitter import BaseInstEmitter, register_emitter
from tilus.hidet.ir.primitives.cuda.mapa import mapa_shared
from tilus.ir.instructions.cuda.mapa import MapSharedAddrInst


@register_emitter(MapSharedAddrInst)
class MapSharedAddrEmitter(BaseInstEmitter):
    def emit(self, inst: MapSharedAddrInst) -> None:
        addr_tensor = inst.register_input
        out_tensor = inst.register_output
        addr_var = self.get_or_allocate_var(addr_tensor)
        out_var = self.get_or_allocate_var(out_tensor)
        with self.for_range(addr_tensor.local_size) as i:
            self.buffer_store(buf=out_var, indices=[i], value=mapa_shared(addr_var[i], inst.target_rank))
