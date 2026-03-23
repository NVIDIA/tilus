# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import namedtuple

from tilus.hidet.ir.dtypes import int32
from tilus.hidet.ir.expr import Var
from tilus.hidet.ir.primitives.vars import lookup_primitive_variable, register_primitive_variable
from tilus.hidet.utils.py import initialize


@initialize()
def register_cuda_primitive_variables():
    for base in ["threadIdx", "blockIdx", "blockDim", "gridDim"]:
        for suffix in ["x", "y", "z"]:
            name = "{}.{}".format(base, suffix)
            register_primitive_variable(name=name, dtype=int32)


def thread_idx(dim="x") -> Var:
    return lookup_primitive_variable("threadIdx.{}".format(dim))


def block_idx(dim="x") -> Var:
    return lookup_primitive_variable("blockIdx.{}".format(dim))


def block_dim(dim="x") -> Var:
    return lookup_primitive_variable("blockDim.{}".format(dim))


def grid_dim(dim="x") -> Var:
    return lookup_primitive_variable("gridDim.{}".format(dim))


dim3 = namedtuple("dim3", field_names=["x", "y", "z"])
threadIdx = dim3(thread_idx("x"), thread_idx("y"), thread_idx("z"))
blockIdx = dim3(block_idx("x"), block_idx("y"), block_idx("z"))
blockDim = dim3(block_dim("x"), block_dim("y"), block_dim("z"))
gridDim = dim3(grid_dim("x"), grid_dim("y"), grid_dim("z"))


# ---------------------------------------------------------------------------
# Cluster variables (from extensions)
# ---------------------------------------------------------------------------


class Dim3:
    def __init__(self, x: Var, y: Var, z: Var):
        self.x: Var = x
        self.y: Var = y
        self.z: Var = z

    def __repr__(self):
        return f"Dim3(x={self.x}, y={self.y}, z={self.z})"

    def __iter__(self):
        return iter((self.x, self.y, self.z))


@initialize()
def register_cuda_cluster_primitive_variables():
    for base in ["clusterBlockIdx", "clusterBlockRank", "clusterDim", "clusterSize", "clusterIdx"]:
        if base == "clusterBlockRank" or base == "clusterSize":
            register_primitive_variable(name=base, dtype=int32)
        else:
            for suffix in ["x", "y", "z"]:
                name = "{}_{}".format(base, suffix)
                register_primitive_variable(name=name, dtype=int32)


def cluster_block_idx(dim: str) -> Var:
    return lookup_primitive_variable("clusterBlockIdx_{}".format(dim))


def cluster_block_rank() -> Var:
    return lookup_primitive_variable("clusterBlockRank")


def cluster_idx(dim: str) -> Var:
    return lookup_primitive_variable("clusterIdx_{}".format(dim))


def cluster_dim(dim: str) -> Var:
    return lookup_primitive_variable("clusterDim_{}".format(dim))


def cluster_size() -> Var:
    return lookup_primitive_variable("clusterSize")


clusterBlockIdx = Dim3(cluster_block_idx("x"), cluster_block_idx("y"), cluster_block_idx("z"))
clusterBlockRank = cluster_block_rank()
clusterDim = Dim3(cluster_dim("x"), cluster_dim("y"), cluster_dim("z"))
clusterIdx = Dim3(cluster_idx("x"), cluster_idx("y"), cluster_idx("z"))
clusterSize = cluster_size()
