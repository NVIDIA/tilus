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
        asm("mbarrier.init.shared::cta.b64 [%0], %1;", inputs=[mbarrier_addr, arrive_count], is_volatile=True)

    @no_type_check
    @script
    def cuda_mbarrier_wait_shared(mbarrier_addr: u32, phase: u32):
        attrs.func_kind = "cuda_internal"
        ticks = u32(10_000_000)
        asm(
            template="{ .reg.pred P1; LAB_WAIT: mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; @!P1 bra.uni LAB_WAIT; }",
            inputs=[mbarrier_addr, phase, ticks],
            is_volatile=True,
        )

    @no_type_check
    @script
    def cuda_mbarrier_arrive_cta_shared(mbarrier_addr: u32):
        attrs.func_kind = "cuda_internal"
        asm(template="mbarrier.arrive.shared::cta.b64 _, [%0];", inputs=[mbarrier_addr], is_volatile=True)

    @no_type_check
    @script
    def cuda_mbarrier_arrive_cluster_shared(mbarrier_addr: u32, cta_id: u32, pred: u32):
        attrs.func_kind = "cuda_internal"
        asm(
            template="{ .reg.pred p; .reg.b32 remAddr32; setp.eq.u32 p, %2, 1; @p mapa.shared::cluster.u32 remAddr32, %0, %1; @p mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32]; }",
            inputs=[mbarrier_addr, cta_id, pred],
            is_volatile=True,
        )

    @no_type_check
    @script
    def cuda_mbarrier_expect_tx_cta_shared(mbarrier_addr: u32, transaction_bytes: u32):
        attrs.func_kind = "cuda_internal"
        asm(
            template="mbarrier.expect_tx.shared::cta.b64 [%0], %1;",
            inputs=[mbarrier_addr, transaction_bytes],
            is_volatile=True,
        )

    @no_type_check
    @script
    def cuda_mbarrier_expect_tx_cluster_shared(mbarrier_addr: u32, transaction_bytes: u32):
        attrs.func_kind = "cuda_internal"
        asm(
            template="mbarrier.expect_tx.relaxed.cluster.shared::cta.b64 [%0], %1;",
            inputs=[mbarrier_addr, transaction_bytes],
            is_volatile=True,
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
        )

    for func in [
        cuda_mbarrier_init_shared,
        cuda_mbarrier_wait_shared,
        cuda_mbarrier_arrive_cta_shared,
        cuda_mbarrier_arrive_cluster_shared,
        cuda_mbarrier_expect_tx_cta_shared,
        cuda_mbarrier_expect_tx_cluster_shared,
        cuda_mbarrier_sync,
    ]:
        register_primitive_function(name=func.name, func_or_type=func)


def mbarrier_init_shared(mbarrier_addr: Expr, arrive_count: Union[int, Expr]) -> Expr:
    return call_primitive_func("cuda_mbarrier_init_shared", args=[mbarrier_addr, u32(arrive_count)])


def mbarrier_wait_shared(mbarrier_addr: Expr, phase: Union[int, Expr]) -> Expr:
    return call_primitive_func("cuda_mbarrier_wait_shared", args=[mbarrier_addr, u32(phase)])


def mbarrier_arrive_cta_shared(mbarrier_addr: Expr) -> Expr:
    return call_primitive_func("cuda_mbarrier_arrive_cta_shared", args=[mbarrier_addr])


def mbarrier_arrive_cluster_shared(mbarrier_addr: Expr, cta_id: Union[int, Expr], pred: Union[bool, Expr]) -> Expr:
    return call_primitive_func("cuda_mbarrier_arrive_cluster_shared", args=[mbarrier_addr, u32(cta_id), boolean(pred)])


def mbarrier_sync_shared(mbarrier_addr: Expr) -> Expr:
    return call_primitive_func("cuda_mbarrier_sync", args=[mbarrier_addr])


def mbarrier_expect_tx_cta_shared(mbarrier_addr: Expr, transaction_bytes: Union[int, Expr]) -> Expr:
    if isinstance(transaction_bytes, int):
        transaction_bytes = u32(transaction_bytes)
    assert isinstance(transaction_bytes, Expr)
    return call_primitive_func("cuda_mbarrier_expect_tx_cta_shared", args=[mbarrier_addr, transaction_bytes])


def mbarrier_expect_tx_cluster_shared(mbarrier_addr: Expr, transaction_bytes: Union[int, Expr]) -> Expr:
    if isinstance(transaction_bytes, int):
        transaction_bytes = u32(transaction_bytes)
    assert isinstance(transaction_bytes, Expr)
    return call_primitive_func("cuda_mbarrier_expect_tx_cluster_shared", args=[mbarrier_addr, transaction_bytes])
