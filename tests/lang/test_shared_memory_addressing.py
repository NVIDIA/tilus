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
import tilus
from tilus import uint8, uint32
from tilus.extensions.hidet.ir.primitives.cuda.cvta import (
    cvta_cluster_shared_to_generic,
    cvta_generic_to_cluster_shared,
    cvta_generic_to_shared,
    cvta_shared_to_generic,
)
from tilus.extensions.hidet.ir.primitives.cuda.mapa import mapa_shared


class SharedMemoryAddressingExample(tilus.Script):
    def __call__(self):
        self.attrs.blocks = (2, 1)
        self.attrs.cluster_blocks = (2, 1)
        self.attrs.warps = 1

        a = self.shared_tensor(dtype=uint8, shape=[1024])
        a_ptr: ~uint8 = ~a[0]
        a_addr_cta: uint32 = cvta_generic_to_shared(a_ptr)
        a_addr_cluster: uint32 = cvta_generic_to_cluster_shared(a_ptr)
        a_addr_cluster_mapa_0: uint32 = mapa_shared(a_addr_cta, cta_rank=1 << 0)
        a_addr_cluster_mapa_1: uint32 = mapa_shared(a_addr_cta, cta_rank=1 << 1)
        a_addr_generic_from_cta: ~uint8 = cvta_shared_to_generic(a_addr_cta)
        a_addr_generic_from_cluster: ~uint8 = cvta_cluster_shared_to_generic(a_addr_cluster)

        self.printf(
            "[%d, %d, %d][%d] Address of a[0]: %p\n",
            self.blockIdx.x,
            self.blockIdx.y,
            self.blockIdx.z,
            self.block_rank_in_cluster,
            a_ptr,
        )
        self.printf(
            "[%d, %d, %d][%d] Address of a[0] (CTA): 0x%x\n",
            self.blockIdx.x,
            self.blockIdx.y,
            self.blockIdx.z,
            self.block_rank_in_cluster,
            a_addr_cta,
        )
        self.printf(
            "[%d, %d, %d][%d] Address of a[0] (Cluster): 0x%x\n",
            self.blockIdx.x,
            self.blockIdx.y,
            self.blockIdx.z,
            self.block_rank_in_cluster,
            a_addr_cluster,
        )
        self.printf(
            "[%d, %d, %d][%d] Address of a[0] (Cluster MAPA at cta_rank=0): 0x%x\n",
            self.blockIdx.x,
            self.blockIdx.y,
            self.blockIdx.z,
            self.block_rank_in_cluster,
            a_addr_cluster_mapa_0,
        )
        self.printf(
            "[%d, %d, %d][%d] Address of a[0] (Cluster MAPA at cta_rank=1): 0x%x\n",
            self.blockIdx.x,
            self.blockIdx.y,
            self.blockIdx.z,
            self.block_rank_in_cluster,
            a_addr_cluster_mapa_1,
        )
        self.printf(
            "[%d, %d, %d][%d] Address of a[0] (Generic from CTA): %p\n",
            self.blockIdx.x,
            self.blockIdx.y,
            self.blockIdx.z,
            self.block_rank_in_cluster,
            a_addr_generic_from_cta,
        )
        self.printf(
            "[%d, %d, %d][%d] Address of a[0] (Generic from Cluster): %p\n",
            self.blockIdx.x,
            self.blockIdx.y,
            self.blockIdx.z,
            self.block_rank_in_cluster,
            a_addr_generic_from_cluster,
        )


def test_shared_memory_address_conversion():
    kernel = SharedMemoryAddressingExample()
    kernel()
