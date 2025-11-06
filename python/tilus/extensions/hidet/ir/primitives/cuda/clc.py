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

from hidet.ir.dtypes import int32, uint32
from hidet.ir.expr import Expr
from hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from hidet.utils import initialize


@initialize()
def register_functions():
    from hidet.lang import asm, attrs, script  # pylint: disable=import-outside-toplevel

    @no_type_check
    @script
    def cuda_cluster_launch_control_try_cancel(mbarrier_addr: uint32, response_smem_addr: uint32, multicast: bool):
        attrs.func_kind = "cuda_internal"

        if multicast:
            asm(
                template="clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];",
                outputs=[],
                inputs=[response_smem_addr, mbarrier_addr],
                is_volatile=True,
            )
        else:
            asm(
                template="clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];",
                outputs=[],
                inputs=[response_smem_addr, mbarrier_addr],
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


def cluster_launch_control_try_cancel(mbarrier: Expr, response: Expr, multicast: Expr | bool) -> Expr:
    """Request cancellation of a cluster that has not launched yet.

    This function requests atomically cancelling the launch of a cluster that has not started running yet.
    It asynchronously writes an opaque 128-bit response to shared memory indicating whether the operation
    succeeded or failed. The completion of the asynchronous operation is tracked using the mbarrier
    completion mechanism at cluster scope.

    On success, the opaque response contains the ctaid of the first CTA of the canceled cluster. No other
    successful response from other clusterlaunchcontrol.try_cancel operations from the same grid will
    contain that id, ensuring uniqueness of the canceled cluster identification.

    The instruction initiates the cancellation operation asynchronously and control returns to the
    executing thread before the requested operation is complete. Upon completion, a complete-tx operation
    with completeCount equal to 16 bytes will be performed on the mbarrier object. The executing thread
    can then use mbarrier instructions to wait for completion of the asynchronous operation.

    Parameters
    ----------
    mbarrier: Expr
        The address of the mbarrier object in shared::cta memory to track completion of the operation.
        When the cancellation operation completes, a complete-tx operation with completeCount of 16 bytes
        will be performed on this mbarrier. The mbarrier should be initialized with appropriate expected
        transaction count before calling this function.
        The given mbarrier should be a uint32 indicating the address of the mbarrier in shared::cta space.
    response: Expr
        The naturally aligned address of a 16-byte wide shared memory location where the request's
        response will be written. The response is an opaque 128-bit value that must be decoded using
        `query_response` to determine if the cancellation succeeded and to retrieve the ctaid of the
        first CTA of the canceled cluster.
        It should be a shared tensor of shape (4,) and dtype int32, providing 128 bits (16 bytes) of storage.
        The given response should be a uint32 indicating the address in shared::cta space.
    multicast: Expr or bool
        Whether to use the `.multicast::cluster::all` qualifier. When True, the response is asynchronously
        written using weak async-proxy writes to the corresponding local shared memory address of each CTA
        in the requesting cluster. The completion of the writes to each CTA's local address is signaled via
        a complete-tx operation to the mbarrier object on the shared memory of that CTA.

        When multicast is enabled, the behavior is undefined if any CTA in the cluster has exited.

        Note: The `.multicast::cluster::all` qualifier is only supported on specific architectures
        (sm_100a, sm_101a/sm_110a, sm_120a and their family-specific variants).

    Returns
    -------
    Expr
        An expression representing the primitive function call. This function does not return a value
        directly; instead, results are asynchronously written to the `response` buffer and signaled
        via the `mbarrier`.

    Notes
    -----
    - This is an asynchronous operation. The response is not immediately available after calling this function.
    - You must wait on the mbarrier before querying the response using `query_response`.
    - The cancellation is atomic and best-effort; it will only succeed if the target cluster has not yet
      started running.
    - If the executing CTA has already observed the completion of a clusterlaunchcontrol.try_cancel
      instruction as failed, then the behavior of issuing a subsequent clusterlaunchcontrol.try_cancel
      instruction is undefined.
    - Both `mbarrier` and `response` addresses must be in the .shared::cta state space.
    - The response address must be naturally aligned for 16-byte access.
    - Requires sm_100 or higher (introduced in PTX ISA version 8.6).
    - No synchronization mechanisms other than mbarrier can be used to guarantee completion of this operation.

    See Also
    --------
    query_response : Decode the response to determine cancellation success and extract CTA information
    https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel
    """
    return call_primitive_func("cuda_cluster_launch_control_try_cancel", args=[mbarrier, response, uint32(multicast)])


def cluster_launch_control_query_response(response: Expr, outputs: Expr) -> Expr:
    """Query the response of a clusterlaunchcontrol.try_cancel operation.

    This function decodes the opaque 128-bit response written by the clusterlaunchcontrol.try_cancel
    instruction. After loading the response from try_cancel, this function extracts two pieces of
    information:

    1. Whether the cluster was successfully canceled
    2. The CTA ID of the first CTA in the canceled cluster (if successful)

    The results are written directly to the `outputs` buffer, which the caller can then inspect to
    determine the outcome of the cancellation request.

    Parameters
    ----------
    response: Expr
        The buffer containing the opaque cancel response in shared memory space. It should be a shared
        tensor of shape (4,) and dtype int32, obtained from a previous call to try_cancel.
        The given response should be a uint32 indicating the address of the response buffer in shared space.
        This buffer must have been populated by a completed try_cancel operation (wait on the mbarrier first).
    outputs: Expr
        A pointer to a buffer where the query results will be written. The buffer should have space for
        at least 4 int32 values that will contain:
        - outputs[0]: x coordinate of the first CTA in the canceled cluster (undefined if failed)
        - outputs[1]: y coordinate of the first CTA in the canceled cluster (undefined if failed)
        - outputs[2]: z coordinate of the first CTA in the canceled cluster (undefined if failed)
        - outputs[3]: 1 if cluster was successfully canceled, 0 otherwise

        Note: If the cancellation request failed (outputs[3] == 0), the CTA coordinates in outputs[0:3]
        are undefined and should not be used.

    Returns
    -------
    Expr
        An expression representing the primitive function call. This function does not return a meaningful
        value; instead, the query results are written directly to the `outputs` buffer.

    Notes
    -----
    - You must wait on the mbarrier from try_cancel before calling this function to ensure the response
      is available.
    - If the cancellation request succeeded (outputs[3] == 1), outputs[0:3] contain the x, y, z coordinates
      of the first CTA in the canceled cluster.
    - If the cancellation request failed (outputs[3] == 0), the behavior of the CTA coordinate extraction
      is undefined, and only outputs[3] should be trusted.
    - Requires sm_100 or higher (introduced in PTX ISA version 8.6).
    - The response buffer must be in shared memory and contain a valid response from try_cancel.

    See Also
    --------
    try_cancel : Initiates a cluster cancellation request
    https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-query-cancel
    """
    return call_primitive_func("cuda_cluster_launch_control_query_response", args=[response, outputs])
