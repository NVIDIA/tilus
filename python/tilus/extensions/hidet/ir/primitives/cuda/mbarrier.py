from typing import Optional, Union
from hidet.utils import initialize
from hidet.ir.expr import Constant, Expr
from hidet.ir.stmt import asm
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.ir.primitives.cuda.funcs import call_cuda
from hidet.lang import script, i32, u32, u64, int32, attrs


@initialize()
def register_mbarrier_primitives():
    @script
    def cuda_mbarrier_expect_tx_shared(mbarrier_addr: u32, transaction_bytes: u32):
        attrs.func_name = 'cuda_mbarrier_expect_tx_shared'
        attrs.func_kind = 'cuda_internal'
        asm(template='mbarrier.expect_tx.shared::cta.b64 [%0], %1;', inputs=[mbarrier_addr, transaction_bytes], is_volatile=True)

    assert isinstance(cuda_mbarrier_expect_tx_shared, Function)
    register_primitive_function(
        name=cuda_mbarrier_expect_tx_shared.name, func_or_type=cuda_mbarrier_expect_tx_shared
    )


def mbarrier_expect_tx_shared(mbarrier_addr: Expr, transaction_bytes: Union[int, Expr]):
    if isinstance(transaction_bytes, int):
        transaction_bytes = u32(transaction_bytes)
    return call_primitive_func('cuda_mbarrier_expect_transaction', args=[mbarrier_addr, transaction_bytes])

