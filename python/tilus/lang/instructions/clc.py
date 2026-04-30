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
from tilus.hidet.ir.expr import Expr, Var
from tilus.ir.tensor import RegisterTensor, SharedTensor
from tilus.lang.constructs.structs import Dim3

from .root import InstructionGroup


class ClusterLaunchControlInstructionGroup(InstructionGroup):
    """Cluster Launch Control (CLC) instructions for dynamic work scheduling on Blackwell GPUs.

    CLC enables a running kernel to **cancel clusters that have not yet started**, effectively
    implementing dynamic grid scheduling and work-stealing patterns. A scheduler CTA can
    request cancellation of a pending cluster, and if successful, take over that cluster's
    work.

    The workflow is:

    1. ``try_cancel()`` — asynchronously request cancellation. An opaque 16-byte response is
       written to shared memory, tracked by an mbarrier.
    2. ``mbarrier.wait()`` — wait for the response to arrive.
    3. ``query_response()`` — decode the response to check if cancellation succeeded and
       retrieve the canceled cluster's CTA coordinates.

    If cancellation succeeds, the scheduler can use the returned CTA ID to compute the
    work tile that the canceled cluster would have processed, and execute that work itself.
    If it fails (the cluster already started), the scheduler can retry or proceed with
    other work.

    With ``multicast=True``, the response is broadcast to all CTAs in the requesting cluster,
    so all CTAs can independently query the result.
    """

    def try_cancel(self, response: SharedTensor, mbarrier: Expr | RegisterTensor, multicast: Expr | bool) -> None:
        """Request cancellation of a cluster that has not yet been launched.

        This instruction asynchronously requests the cancellation of a cluster that has not started running yet.
        It writes an opaque 16-byte response to shared memory indicating whether the operation succeeded or failed.
        The completion of the asynchronous operation is tracked using the provided mbarrier.

        On success, the response contains the CTA ID of the first CTA of the canceled cluster. No other successful
        response from other `try_cancel` operations from the same grid will contain that ID.

        The response can be decoded using the `query_response` method to determine if the cancellation was successful
        and to retrieve the CTA ID of the first CTA in the canceled cluster.

        Important: If the executing CTA has already observed the completion of a `try_cancel` instruction as failed,
        then issuing a subsequent `try_cancel` instruction results in undefined behavior.

        Parameters
        ----------
        response: SharedTensor
            A naturally aligned 16-byte wide shared memory tensor where the request's response will be written.
            Must be in .shared::cta state space.
        mbarrier: Expr | RegisterTensor
            The mbarrier object used to track completion of the asynchronous operation. This instruction
            automatically performs an mbarrier arrive operation combined with an expect-tx operation on the
            mbarrier, setting the transaction count to 16 bytes. When the asynchronous write to `response`
            completes, a complete-tx operation with completeCount equal to 16 bytes will be performed on this
            mbarrier, decrementing the tx-count by 16 bytes and potentially allowing the mbarrier to transition
            to the next phase once both tx-count and pending arrivals reach zero.
        multicast: Expr | bool
            If True, the response is asynchronously written using weak async-proxy writes to the corresponding
            local shared memory address of each CTA in the requesting cluster. In multicast mode, for each CTA
            in the cluster, an mbarrier arrive operation combined with an expect-tx operation (16 bytes) is
            performed on that CTA's mbarrier. The completion of the writes to each CTA is signaled via a
            complete-tx operation to the mbarrier object on that CTA's shared memory. When using multicast,
            at least 32 threads are required in the current thread group, and the behavior is undefined if any
            CTA in the cluster has exited.
            If False, a single mbarrier arrive with expect-tx operation is performed on the local mbarrier,
            and the response is written only to the local shared memory of the calling CTA.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group (at least 32 threads if ``multicast=True``).
        - **Hardware**: Requires compute capability 10.0+ (sm_100).
        - **PTX**: ``clusterlaunchcontrol.try_cancel``
        - This instruction performs an mbarrier arrive operation combined with an expect-tx operation (16 bytes)
          before issuing the cancellation request. The mbarrier's tx-count is increased by 16 bytes when the
          instruction is issued, and decreased by 16 bytes when the response write completes asynchronously.
        """
        self._builder.cluster_launch_control_try_cancel(response, mbarrier, multicast)

    def query_response(self, response: SharedTensor) -> tuple[Var, Dim3]:
        """Query the response from a cluster launch control try_cancel operation.

        This instruction decodes the opaque 16-byte response written by the `try_cancel` instruction. It extracts
        two pieces of information:
        1. Whether the cancellation request succeeded (is_canceled)
        2. If successful, the CTA ID (x, y, z coordinates) of the first CTA in the canceled cluster

        The response should be loaded from shared memory after the mbarrier used in `try_cancel` has signaled
        completion of the asynchronous operation.

        If the cancellation request failed, the CTA ID coordinates in the returned Dim3 are undefined and should
        not be used.

        Parameters
        ----------
        response: SharedTensor
            A 16-byte wide shared memory tensor containing the opaque response from a `try_cancel` operation.
            This should be the same tensor that was passed to `try_cancel`.

        Returns
        -------
        is_canceled: Var
            A variable (predicate/boolean) that is True if the cluster was successfully canceled, False otherwise.
        first_cta_id: Dim3
            If the cancellation succeeded, this contains the (x, y, z) coordinates of the first CTA in the
            canceled cluster. If the cancellation failed, these values are undefined.

        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 10.0+ (sm_100).
        - **PTX**: ``clusterlaunchcontrol.query_cancel.is_canceled`` and
          ``clusterlaunchcontrol.query_cancel.get_first_ctaid``
        - The behavior is undefined if called before the ``try_cancel`` operation has completed (i.e., before
          the associated mbarrier has signaled completion).
        """
        ret = self._builder.cluster_launch_control_query_response(response)
        items = []
        for i in range(4):  # (is_canceled, first_cta_x, first_cta_y, first_cta_z)
            items.append(
                self._builder.tensor_item_value(
                    self._builder.slice_register(ret, offsets=[i], slice_dims=[], slice_shape=[])
                )
            )
        return (items[0], Dim3(items[1], items[2], items[3]))
