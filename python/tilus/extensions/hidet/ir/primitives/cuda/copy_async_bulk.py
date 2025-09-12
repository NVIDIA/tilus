from typing import Optional, no_type_check

from hidet.ir.dtypes import int32, uint32, uint16
from hidet.ir.expr import Expr
from hidet.ir.func import Function
from hidet.ir.primitives import register_primitive_function
from hidet.ir.primitives.cuda.funcs import call_cuda
from hidet.ir.type import void_p
from hidet.utils import initialize


def resolve_cp_async_bulk_global_to_shared(l2_evict: Optional[str]) -> str:
    if l2_evict not in [None, "evict_first"]:
        raise ValueError("l2_evict should be None or 'evict_first'")
    func_name = "cp_async_bulk_shared_global{}".format("_l2_evict_first" if l2_evict == "evict_first" else "")
    return func_name

def resolve_cp_async_bulk_global_to_cluster_shared(l2_evict: Optional[str]) -> str:
    if l2_evict not in [None, "evict_first"]:
        raise ValueError("l2_evict should be None or 'evict_first'")
    func_name = "cp_async_bulk_cluster_shared_global{}".format("_l2_evict_first" if l2_evict == "evict_first" else "")
    return func_name


def resolve_cp_async_bulk_shared_to_global(l2_evict: Optional[str], cp_mask: bool) -> str:
    if l2_evict not in [None, "evict_first"]:
        raise ValueError("l2_evict should be None or 'evict_first'")
    func_name = "cp_async_bulk_global_shared{}".format("_l2_evict_first" if l2_evict == "evict_first" else "")
    if cp_mask:
        func_name += "_cp_mask"
    return func_name


@initialize()
def register_bulk_copy_async():
    from hidet.lang import asm, attrs, script

    # cp_async_bulk_global_to_shared
    for l2_evict in [None, "evict_first"]:
        func_name = resolve_cp_async_bulk_global_to_shared(l2_evict)
        if l2_evict == "evict_first":
            template_string = (
                "{"
                "    .reg .b64 p;"
                "    createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
                "    cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], p;"
                "}"
            )
        else:
            template_string = (
                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
            )
        func_name = "cuda_" + func_name

        @no_type_check
        @script
        def cuda_cp_async(dst: uint32, src: void_p, size: int32, mbarrier: uint32):
            attrs.func_name = func_name
            attrs.func_kind = "cuda_internal"
            asm(template=template_string, inputs=[dst, src, size, mbarrier], is_volatile=True, memory_fence=True)

        assert isinstance(cuda_cp_async, Function)
        register_primitive_function(name=func_name, func_or_type=cuda_cp_async)

    # cp_async_bulk_global_to_cluster_shared
    for l2_evict in [None, "evict_first"]:
        func_name = resolve_cp_async_bulk_global_to_cluster_shared(l2_evict)
        if l2_evict == "evict_first":
            template_string = (
                "{"
                "    .reg .b64 p;"
                "    createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
                "    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1], %2, [%3], %4, p;"
                "}"
            )
        else:
            template_string = (
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
            )
        func_name = "cuda_" + func_name

        @no_type_check
        @script
        def cuda_cp_async(dst: uint32, src: void_p, size: int32, mbarrier: uint32, cta_mask: uint16):
            attrs.func_name = func_name
            attrs.func_kind = "cuda_internal"
            asm(template=template_string, inputs=[dst, src, size, mbarrier, cta_mask], is_volatile=True, memory_fence=True)

        assert isinstance(cuda_cp_async, Function)
        register_primitive_function(name=func_name, func_or_type=cuda_cp_async)

    # cp_async_bulk_s2g
    for l2_evict in [None, "evict_first"]:
        for cp_mask in [False, True]:
            func_name = resolve_cp_async_bulk_shared_to_global(l2_evict, cp_mask)

            operand_count = 3
            inst = "cp.async.bulk.global.shared::cta.bulk_group{cache_hint}{cp_mask} [%0], [%1], %2".format(
                cache_hint=".L2::cache_hint" if l2_evict is not None else "",
                cp_mask=".cp_mask" if cp_mask else "",
            )
            if l2_evict is not None:
                inst = inst + ", p"
            if cp_mask:
                inst = inst + ", %{}".format(operand_count)
                operand_count += 1
            if l2_evict == "evict_first":
                template_string = (
                    "{"
                    "    .reg .b64 p;"
                    "    createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
                    "    " + inst + ";"
                    "}"
                )
            else:
                template_string = inst + ";"

            func_name = "cuda_" + func_name
            if not cp_mask:

                @no_type_check
                @script
                def cuda_cp_async(dst: void_p, src: uint32, size: int32):
                    attrs.func_name = func_name
                    attrs.func_kind = "cuda_internal"
                    inputs = [dst, src, size]
                    asm(template=template_string, inputs=inputs, is_volatile=True, memory_fence=True)
            else:

                @no_type_check
                @script
                def cuda_cp_async(dst: void_p, src: uint32, size: int32, byte_mask: uint32):
                    attrs.func_name = func_name
                    attrs.func_kind = "cuda_internal"
                    inputs = [dst, src, size, byte_mask]
                    asm(template=template_string, inputs=inputs, is_volatile=True, memory_fence=True)

            assert isinstance(cuda_cp_async, Function)
            register_primitive_function(name=func_name, func_or_type=cuda_cp_async)

    # cp_async_shared_to_cluster_shared
    func_name = "cuda_cp_async_bulk_cluster_shared_shared"
    @no_type_check
    @script
    def cuda_cp_async(dst: uint32, src: uint32, size: int32, mbarrier: uint32):
        attrs.func_name = func_name
        attrs.func_kind = "cuda_internal"
        asm(
            template="cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];",
            inputs=[dst, src, size, mbarrier],
            is_volatile=True,
            memory_fence=True,
        )
    assert isinstance(cuda_cp_async, Function)
    register_primitive_function(name=func_name, func_or_type=cuda_cp_async)




def cp_async_bulk_global_to_shared(dst: Expr, src: Expr, size: Expr, mbarrier: Expr, l2_evict: Optional[str] = None) -> Expr:
    """Perform a bulk copy from global memory to shared memory asynchronously.

    Parameters
    ----------
    dst: Expr
        The destination address in shared memory. It should be an address with shared memory space with type uint32.
    src: Expr
        The source address in global memory. It should be an address with global memory space with type void_p.
    size: Expr
        The size of the data to be copied in bytes. It should be an expression with int32 type. It must be a multiple
        of 16.
    mbarrier: Expr
        The mbarrier to be used for synchronization. It should be an address with shared memory space with type uint32
        that has been initialized by `mbarrier_init`.
    l2_evict: str, optional
        The L2 cache eviction policy. It can be:
        - None: the default policy.
        - 'evict_first': mark the cached data caused by this copy as `evict_first` in L2 cache. Used for data that will
          not be reused.
    Returns
    -------
    ret: Expr
        A function call expression.
    """
    func_name = resolve_cp_async_bulk_global_to_shared(l2_evict)
    return call_cuda(func_name, args=[dst, src, size, mbarrier])


def cp_async_bulk_global_to_cluster_shared(dst: Expr, src: Expr, size: Expr, mbarrier: Expr, cta_mask: int, l2_evict: Optional[str] = None) -> Expr:
    """Perform a bulk copy from global memory to cluster's CTA shared memory asynchronously.

    Parameters
    ----------
    dst: Expr
        The destination address in shared memory. It should be an address with shared memory space with type uint32.
    src: Expr
        The source address in global memory. It should be an address with global memory space with type void_p.
    size: Expr
        The size of the data to be copied in bytes. It should be an expression with int32 type. It must be a multiple
        of 16.
    mbarrier: Expr
        The mbarrier to be used for synchronization. It should be an address with shared memory space with type uint32
        that has been initialized by `mbarrier_init`.
    cta_mask: int
        The CTA mask to specify which CTAs in the cluster will perform the copy. Each bit in the mask corresponds to a
        CTA in the cluster. If the bit is 1, the corresponding CTA will perform the copy; otherwise, it will not.
    l2_evict: str, optional
        The L2 cache eviction policy. It can be:
        - None: the default policy.
        - 'evict_first': mark the cached data caused by this copy as `evict_first` in L2 cache. Used for data that will
          not be reused.
    Returns
    -------
    ret: Expr
        A function call expression.
    """
    func_name = resolve_cp_async_bulk_global_to_cluster_shared(l2_evict)
    return call_cuda(func_name, args=[dst, src, size, mbarrier, cta_mask])


def cp_async_bulk_shared_to_global(
    dst: Expr,
    src: Expr,
    size: Expr,
    l2_evict: Optional[str] = None,
    byte_mask: Optional[Expr] = None,
) -> Expr:
    """Perform a bulk copy from shared memory to global memory asynchronously.

    Parameters
    ----------
    dst: Expr
        The destination address in global memory. It should be an address with global memory space with type void_p.
    src: Expr
        The source address in shared memory. It should be an address with shared memory space with type uint32.
    size: Expr
        The size of the data to be copied in bytes. It should be an expression with int32 type. It must be a multiple
        of 16.
    l2_evict: str, optional
        The L2 cache eviction policy. It can be:
        - None: the default policy.
        - 'evict_first': mark the cached data caused by this copy as `evict_first` in L2 cache. Used for data that will
          not be reused.
    byte_mask: Expr, optional
        A byte mask to specify which bytes to be copied. It should be an expression with uint32 type. If it is None,
        all bytes will be copied. We can split the copy into multiple 16-byte segments, and each byte in the segment
        corresponds to a bit in the byte mask. If the bit is 1, the corresponding byte will be copied; otherwise, it
        will not be copied.

    Returns
    -------
    ret: Expr
        A function call expression.
    """
    func_name = resolve_cp_async_bulk_shared_to_global(l2_evict, byte_mask is not None)
    args = [dst, src, size]
    if byte_mask is not None:
        args.append(byte_mask)
    return call_cuda(func_name, args=args)


def cp_async_bulk_shared_to_cluster_shared(
    dst: Expr,
    src: Expr,
    size: Expr,
    mbarrier: Expr,
) -> Expr:
    """Perform a bulk copy from cluster's CTA shared memory to shared memory asynchronously.

    Parameters
    ----------
    dst: Expr
        The destination address in shared memory. It should be an address with shared memory space with type uint32.
    src: Expr
        The source address in shared memory. It should be an address with shared memory space with type uint32.
    size: Expr
        The size of the data to be copied in bytes. It should be an expression with int32 type. It must be a multiple
        of 16.
    mbarrier: Expr
        The mbarrier to be used for synchronization. It should be an address with shared memory space with type uint32
        that has been initialized by `mbarrier_init`.

    Returns
    -------
    ret: Expr
        A function call expression.
    """
    func_name = "cp_async_bulk_cluster_shared_shared"
    return call_cuda(func_name, args=[dst, src, size, mbarrier])
