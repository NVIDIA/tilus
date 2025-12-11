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
from typing import Union, no_type_check

from hidet.ir.dtypes import boolean
from hidet.ir.expr import Expr
from hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from hidet.ir.stmt import asm
from hidet.lang import attrs, script, u32
from hidet.utils import initialize


@initialize()
def register_mbarrier_primitives():
    @no_type_check
    @script
    def cuda_mbarrier_init_shared(mbarrier_addr: u32, arrive_count: u32):
        attrs.func_kind = "cuda_internal"
        asm(
            "mbarrier.init.shared::cta.b64 [%0], %1;",
            inputs=[mbarrier_addr, arrive_count],
            is_volatile=True,
            memory_fence=True,
        )

    @no_type_check
    @script
    def cuda_mbarrier_wait_shared(mbarrier_addr: u32, phase: u32):
        attrs.func_kind = "cuda_internal"
        ticks = u32(10_000_000)
        asm(
            template="{ .reg.pred P1; LAB_WAIT: mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; @!P1 bra.uni LAB_WAIT; }",
            inputs=[mbarrier_addr, phase, ticks],
            is_volatile=True,
            memory_fence=True,
        )

    @no_type_check
    @script
    def cuda_mbarrier_arrive_shared(mbarrier_addr: u32, count: u32):
        attrs.func_kind = "cuda_internal"
        asm(template="mbarrier.arrive.shared::cta.b64 _, [%0], %1;", inputs=[mbarrier_addr, count], is_volatile=True)

    @no_type_check
    @script
    def cuda_mbarrier_arrive_remote_shared(mbarrier_addr: u32, count: u32, cta_id: u32, pred: u32):
        attrs.func_kind = "cuda_internal"
        asm(
            template="{ .reg.pred p; .reg.b32 remAddr32; setp.ne.u32 p, %3, 0; @p mapa.shared::cluster.u32 remAddr32, %0, %2; @p mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32], %1; }",
            inputs=[mbarrier_addr, count, cta_id, pred],
            is_volatile=True,
            memory_fence=True,
        )

    @no_type_check
    @script
    def cuda_mbarrier_expect_tx_shared(mbarrier_addr: u32, transaction_bytes: u32):
        attrs.func_kind = "cuda_internal"
        asm(
            template="mbarrier.expect_tx.shared::cta.b64 [%0], %1;",
            inputs=[mbarrier_addr, transaction_bytes],
            is_volatile=True,
            memory_fence=True,
        )

    @no_type_check
    @script
    def cuda_mbarrier_expect_tx_remote_shared(mbarrier_addr: u32, transaction_bytes: u32, cta_id: u32, pred: u32):
        attrs.func_kind = "cuda_internal"
        asm(
            template="{ .reg.pred p; .reg.b32 remAddr32; setp.ne.u32 p, %3, 0; @p mapa.shared::cluster.u32 remAddr32, %0, %2; @p mbarrier.expect_tx.relaxed.cluster.shared::cluster.b64 [remAddr32], %1; }",
            inputs=[mbarrier_addr, transaction_bytes, cta_id, pred],
            is_volatile=True,
            memory_fence=True,
        )

    @no_type_check
    @script
    def cuda_mbarrier_arrive_and_expect_tx_shared(mbarrier_addr: u32, transaction_bytes: u32):
        attrs.func_kind = "cuda_internal"
        asm(
            template="mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;",
            inputs=[mbarrier_addr, transaction_bytes],
            is_volatile=True,
            memory_fence=True,
        )

    @no_type_check
    @script
    def cuda_mbarrier_arrive_and_expect_tx_remote_shared(
        mbarrier_addr: u32, transaction_bytes: u32, cta_id: u32, pred: u32
    ):
        attrs.func_kind = "cuda_internal"
        asm(
            template="{ .reg.pred p; .reg.b32 remAddr32; setp.ne.u32 p, %3, 0; @p mapa.shared::cluster.u32 remAddr32, %0, %2; @p mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [remAddr32], %1; }",
            inputs=[mbarrier_addr, transaction_bytes, cta_id, pred],
            is_volatile=True,
            memory_fence=True,
        )

    @no_type_check
    @script
    def cuda_mbarrier_sync(mbarrier_addr: u32):
        attrs.func_kind = "cuda_internal"
        ticks = u32(10_000_000)
        asm(
            template="{ "
            "  .reg.pred P1; "
            "  .reg.b64 state; "
            "  mbarrier.arrive.shared::cta.b64 state, [%0]; "
            "  LAB_WAIT: mbarrier.try_wait.shared::cta.b64 P1, [%0], state, %1; "
            "  @!P1 bra.uni LAB_WAIT; "
            "}",
            inputs=[mbarrier_addr, ticks],
            is_volatile=True,
            memory_fence=True,
        )

    for func in [
        cuda_mbarrier_init_shared,
        cuda_mbarrier_wait_shared,
        cuda_mbarrier_arrive_shared,
        cuda_mbarrier_arrive_remote_shared,
        cuda_mbarrier_expect_tx_shared,
        cuda_mbarrier_expect_tx_remote_shared,
        cuda_mbarrier_arrive_and_expect_tx_shared,
        cuda_mbarrier_arrive_and_expect_tx_remote_shared,
        cuda_mbarrier_sync,
    ]:
        register_primitive_function(name=func.name, func_or_type=func)


def mbarrier_init_shared(mbarrier_addr: Expr, arrive_count: Union[int, Expr]) -> Expr:
    """
    Initialize an mbarrier object in shared memory.

    Initializes the mbarrier object with the specified expected arrival count. The mbarrier object
    must be 8-byte aligned and located in shared memory. After initialization, the current phase
    is set to 0, and both expected and pending arrival counts are set to `arrive_count`.

    Parameters
    ----------
    mbarrier_addr : Expr
        Address of the mbarrier object in shared memory (u32).
    arrive_count : Union[int, Expr]
        Number of expected arrivals per phase. Valid range is [1, 2^20 - 1].

    Returns
    -------
    ret : Expr
        A call expression that performs the initialization.

    See Also
    --------
    mbarrier.init : PTX ISA documentation section 9.7.13.15.9
    """
    return call_primitive_func("cuda_mbarrier_init_shared", args=[mbarrier_addr, u32(arrive_count)])


def mbarrier_wait_shared(mbarrier_addr: Expr, phase: Union[int, Expr]) -> Expr:
    """
    Wait for the specified phase of an mbarrier object to complete.

    Blocks execution until the mbarrier completes the phase indicated by `phase` parity.
    Uses try_wait internally with a timeout hint of 10,000,000 nanoseconds.

    Parameters
    ----------
    mbarrier_addr : Expr
        Address of the mbarrier object in shared memory (u32).
    phase : Union[int, Expr]
        Phase parity to wait for (0 for even phases, 1 for odd phases).

    Returns
    -------
    ret : Expr
        A call expression that performs the wait operation.

    See Also
    --------
    mbarrier.try_wait.parity : PTX ISA documentation section 9.7.13.15.16
    """
    return call_primitive_func("cuda_mbarrier_wait_shared", args=[mbarrier_addr, u32(phase)])


def mbarrier_arrive_shared(mbarrier_addr: Expr, count: Expr | int) -> Expr:
    """
    Perform an arrive operation on an mbarrier object in shared memory.

    Signals the arrival of the executing thread by decrementing the pending arrival count
    by `count`. If all arrivals and transactions are complete, the mbarrier transitions
    to the next phase.

    Parameters
    ----------
    mbarrier_addr : Expr
        Address of the mbarrier object in shared memory (u32).
    count : Union[int, Expr]
        Number of arrivals to signal. Defaults to 1 if not specified.

    Returns
    -------
    ret : Expr
        A call expression that performs the arrive operation.

    See Also
    --------
    mbarrier.arrive : PTX ISA documentation section 9.7.13.15.13
    """
    count_expr = count if isinstance(count, Expr) else u32(count)
    return call_primitive_func("cuda_mbarrier_arrive_shared", args=[mbarrier_addr, count_expr])


def mbarrier_arrive_remote_shared(
    mbarrier_addr: Expr, count: Expr | int, cta_id: Union[int, Expr], pred: Union[bool, Expr]
) -> Expr:
    """
    Perform an arrive operation on a remote mbarrier object in cluster shared memory.

    Maps the local shared memory address to a remote CTA's address space and performs
    an arrive operation with release semantics at cluster scope. The operation is
    predicated on `pred`.

    Parameters
    ----------
    mbarrier_addr : Expr
        Address of the mbarrier object in local shared memory (u32).
    count : Union[int, Expr]
        Number of arrivals to signal.
    cta_id : Union[int, Expr]
        Target CTA ID within the cluster.
    pred : Union[bool, Expr]
        Predicate controlling whether the operation is performed.

    Returns
    -------
    ret : Expr
        A call expression that performs the remote arrive operation.

    See Also
    --------
    mbarrier.arrive : PTX ISA documentation section 9.7.13.15.13
    """
    return call_primitive_func(
        "cuda_mbarrier_arrive_remote_shared", args=[mbarrier_addr, u32(count), u32(cta_id), boolean(pred)]
    )


def mbarrier_sync_shared(mbarrier_addr: Expr) -> Expr:
    """
    Perform a combined arrive and wait operation on an mbarrier object.

    Convenience function that performs an arrive operation and immediately waits for
    the phase to complete using try_wait. Equivalent to calling arrive followed by
    a wait loop.

    Parameters
    ----------
    mbarrier_addr : Expr
        Address of the mbarrier object in shared memory (u32).

    Returns
    -------
    ret : Expr
        A call expression that performs the synchronization.

    See Also
    --------
    mbarrier.arrive : PTX ISA documentation section 9.7.13.15.13
    mbarrier.try_wait : PTX ISA documentation section 9.7.13.15.16
    """
    return call_primitive_func("cuda_mbarrier_sync", args=[mbarrier_addr])


def mbarrier_expect_tx_shared(mbarrier_addr: Expr, transaction_bytes: Union[int, Expr]) -> Expr:
    """
    Increase the transaction count of an mbarrier object in shared memory.

    Performs an expect-tx operation that increases the tx-count by `transaction_bytes`.
    This sets the current phase to track additional asynchronous memory transactions.
    The tx-count must reach zero along with pending arrivals for phase completion.

    Parameters
    ----------
    mbarrier_addr : Expr
        Address of the mbarrier object in shared memory (u32).
    transaction_bytes : Union[int, Expr]
        Number of transaction bytes to expect, in units specified by async operations.

    Returns
    -------
    ret : Expr
        A call expression that performs the expect-tx operation.

    See Also
    --------
    mbarrier.expect_tx : PTX ISA documentation section 9.7.13.15.11
    """
    return call_primitive_func("cuda_mbarrier_expect_tx_shared", args=[mbarrier_addr, u32(transaction_bytes)])


def mbarrier_expect_tx_remote_shared(
    mbarrier_addr: Expr, transaction_bytes: Union[int, Expr], cta_id: Union[int, Expr], pred: Union[bool, Expr]
) -> Expr:
    """
    Increase the transaction count of a remote mbarrier object in cluster shared memory.

    Maps the local shared memory address to a remote CTA's address space and performs
    an expect-tx operation with relaxed semantics at cluster scope. The operation is
    predicated on `pred`.

    Parameters
    ----------
    mbarrier_addr : Expr
        Address of the mbarrier object in local shared memory (u32).
    transaction_bytes : Union[int, Expr]
        Number of transaction bytes to expect.
    cta_id : Union[int, Expr]
        Target CTA ID within the cluster.
    pred : Union[bool, Expr]
        Predicate controlling whether the operation is performed.

    Returns
    -------
    ret : Expr
        A call expression that performs the remote expect-tx operation.

    See Also
    --------
    mbarrier.expect_tx : PTX ISA documentation section 9.7.13.15.11
    """
    return call_primitive_func(
        "cuda_mbarrier_expect_tx_remote_shared",
        args=[mbarrier_addr, u32(transaction_bytes), u32(cta_id), boolean(pred)],
    )


def mbarrier_arrive_and_expect_tx_shared(mbarrier_addr: Expr, transaction_bytes: Union[int, Expr]) -> Expr:
    """
    Perform combined expect-tx and arrive operations on an mbarrier object.

    Atomically performs an expect-tx operation followed by an arrive operation with
    count=1. This is equivalent to calling expect_tx then arrive, but as a single
    atomic operation with release semantics.

    Parameters
    ----------
    mbarrier_addr : Expr
        Address of the mbarrier object in shared memory (u32).
    transaction_bytes : Union[int, Expr]
        Number of transaction bytes to expect.

    Returns
    -------
    ret : Expr
        A call expression that performs the combined operation.

    See Also
    --------
    mbarrier.arrive.expect_tx : PTX ISA documentation section 9.7.13.15.13
    """
    return call_primitive_func(
        "cuda_mbarrier_arrive_and_expect_tx_shared", args=[mbarrier_addr, u32(transaction_bytes)]
    )


def mbarrier_arrive_and_expect_tx_remote_shared(
    mbarrier_addr: Expr, transaction_bytes: Union[int, Expr], cta_id: Union[int, Expr], pred: Union[bool, Expr]
) -> Expr:
    """
    Perform combined expect-tx and arrive operations on a remote mbarrier object.

    Maps the local shared memory address to a remote CTA's address space and performs
    an atomic expect-tx followed by arrive operation with release semantics at cluster
    scope. The operation is predicated on `pred`.

    Parameters
    ----------
    mbarrier_addr : Expr
        Address of the mbarrier object in local shared memory (u32).
    transaction_bytes : Union[int, Expr]
        Number of transaction bytes to expect.
    cta_id : Union[int, Expr]
        Target CTA ID within the cluster.
    pred : Union[bool, Expr]
        Predicate controlling whether the operation is performed.

    Returns
    -------
    ret : Expr
        A call expression that performs the remote combined operation.

    See Also
    --------
    mbarrier.arrive.expect_tx : PTX ISA documentation section 9.7.13.15.13
    """
    return call_primitive_func(
        "cuda_mbarrier_arrive_and_expect_tx_remote_shared",
        args=[mbarrier_addr, u32(transaction_bytes), u32(cta_id), boolean(pred)],
    )
