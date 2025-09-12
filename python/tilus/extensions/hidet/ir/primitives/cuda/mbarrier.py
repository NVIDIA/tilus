from typing import Union, no_type_check

from hidet.ir.expr import Expr
from hidet.ir.primitives.func import call_primitive_func, register_primitive_function
from hidet.ir.stmt import asm
from hidet.lang import attrs, script, u32
from hidet.utils import initialize


@initialize()
def register_mbarrier_primitives():
    @no_type_check
    @script
    def cuda_mbarrier_expect_tx_cta_shared(mbarrier_addr: u32, transaction_bytes: u32):
        attrs.func_name = "cuda_mbarrier_expect_tx_cta_shared"
        attrs.func_kind = "cuda_internal"
        asm(
            template="mbarrier.expect_tx.shared::cta.b64 [%0], %1;",
            inputs=[mbarrier_addr, transaction_bytes],
            is_volatile=True,
        )

    @no_type_check
    @script
    def cuda_mbarrier_expect_tx_cluster_shared(mbarrier_addr: u32, transaction_bytes: u32):
        attrs.func_name = "cuda_mbarrier_expect_tx_cluster_shared"
        attrs.func_kind = "cuda_internal"
        asm(
            template="mbarrier.expect_tx.cluster.shared::cta.b64 [%0], %1;",
            inputs=[mbarrier_addr, transaction_bytes],
            is_volatile=True,
        )

    for func in [cuda_mbarrier_expect_tx_cta_shared, cuda_mbarrier_expect_tx_cluster_shared]:
        register_primitive_function(name=func.name, func_or_type=func)


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
