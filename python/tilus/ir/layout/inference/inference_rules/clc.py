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
from tilus import RegisterLayout
from tilus.ir.instructions.cuda.clc import ClusterLaunchControlQueryResponseInst, ClusterLaunchControlTryCancelInst
from tilus.ir.layout import ops
from tilus.ir.layout.inference.rule import LayoutInferenceContext, LayoutInferenceRule, register_rule
from tilus.ir.tensor import RegisterTensor


@register_rule(ClusterLaunchControlTryCancelInst)
class ClusterLaunchControlTryCancelInstRule(LayoutInferenceRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: ClusterLaunchControlTryCancelInst
    ) -> dict[RegisterTensor, RegisterLayout]:
        return {}


@register_rule(ClusterLaunchControlQueryResponseInst)
class ClusterLaunchControlQueryResponseInstRule(LayoutInferenceRule):
    @staticmethod
    def inference(
        ctx: LayoutInferenceContext, inst: ClusterLaunchControlQueryResponseInst
    ) -> dict[RegisterTensor, RegisterLayout]:
        out = inst.register_output
        if out.optional_layout is None:
            return {out: ops.replicated(*out.shape, num_workers=ctx.num_threads)}
        return {}
