# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tilus.hidet.ir.dtypes import int32, uint32
from tilus.hidet.ir.expr import Expr
from tilus.hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from tilus.hidet.utils import initialize


@initialize()
def register_functions():
    from tilus.hidet.lang import asm, attrs, script  # pylint: disable=import-outside-toplevel

    @no_type_check
    @script
    def cuda_cluster_launch_control_try_cancel(
        mbarrier_addr: uint32, response_smem_addr: uint32, multicast: bool, predicate: uint32
    ):
        attrs.func_kind = "cuda_internal"

        if multicast:
            asm(
                template=(
                    "{.reg.pred __pred; setp.ne.u32 __pred, %2, 0;"
                    " @__pred clusterlaunchcontrol.try_cancel.async.shared::cta"
                    ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];}"
                ),
                outputs=[],
                inputs=[response_smem_addr, mbarrier_addr, predicate],
                is_volatile=True,
            )
        else:
            asm(
                template=(
                    "{.reg.pred __pred; setp.ne.u32 __pred, %2, 0;"
                    " @__pred clusterlaunchcontrol.try_cancel.async.shared::cta"
                    ".mbarrier::complete_tx::bytes.b128 [%0], [%1];}"
                ),
                outputs=[],
                inputs=[response_smem_addr, mbarrier_addr, predicate],
                is_volatile=True,
            )

    @no_type_check
    @script
    def cuda_cluster_launch_control_query_response(response_smem_addr: uint32, outputs: ~int32):
        attrs.func_kind = "cuda_internal"

        asm(
            template="{"
            ".reg .pred p1; "
            ".reg .b128 clc_result; "
            "ld.shared.b128 clc_result, [%4]; "
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result; "
            "selp.s32 %3, 1, 0, p1; "
            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_result; "
            "}",
            outputs=[outputs[1], outputs[2], outputs[3], outputs[0]],
            inputs=[response_smem_addr],
            is_volatile=True,
            memory_fence=True,
        )

    for func in [cuda_cluster_launch_control_try_cancel, cuda_cluster_launch_control_query_response]:
        register_primitive_function(name=func.name, func_or_type=func)


def cluster_launch_control_try_cancel(
    mbarrier: Expr, response: Expr, multicast: Expr | bool, predicate: Expr = uint32(1)
) -> Expr:
    """Request cancellation of a cluster that has not launched yet.

    This function requests atomically cancelling the launch of a cluster that has not started running yet.
    It asynchronously writes an opaque 128-bit response to shared memory indicating whether the operation
    succeeded or failed. The completion of the asynchronous operation is tracked using the mbarrier
    completion mechanism at cluster scope.

    Parameters
    ----------
    mbarrier: Expr
        The address of the mbarrier object in shared::cta memory to track completion of the operation.
    response: Expr
        The naturally aligned address of a 16-byte wide shared memory location where the request's
        response will be written.
    multicast: Expr or bool
        Whether to use the `.multicast::cluster::all` qualifier.

    Returns
    -------
    Expr
        An expression representing the primitive function call.
    """
    return call_primitive_func(
        "cuda_cluster_launch_control_try_cancel", args=[mbarrier, response, uint32(multicast), predicate]
    )


def cluster_launch_control_query_response(response: Expr, outputs: Expr) -> Expr:
    """Query the response of a clusterlaunchcontrol.try_cancel operation.

    Parameters
    ----------
    response: Expr
        The buffer containing the opaque cancel response in shared memory space.
    outputs: Expr
        A pointer to a buffer where the query results will be written.

    Returns
    -------
    Expr
        An expression representing the primitive function call.
    """
    return call_primitive_func("cuda_cluster_launch_control_query_response", args=[response, outputs])
