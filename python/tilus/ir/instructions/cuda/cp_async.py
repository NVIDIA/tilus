from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from hidet.ir.expr import Expr, Var, as_expr

from tilus.extensions.hidet.ir.expr import index_vars
from tilus.ir.inst import Instruction
from tilus.ir.tensor import GlobalTensor, SharedTensor


@dataclass(frozen=True, eq=False)
class CopyAsyncInst(Instruction):
    offsets: tuple[Expr, ...]
    dims: Optional[tuple[int, ...]]
    evict: Optional[str]
    weak_mask: bool
    # weak_mask: whether to use the first element of each cp.async instruction as the mask of the whole instruction.
    # By default, weak_mask=False, we require the mask is constant among all elements in the cp.async instruction.
    # However, in some cases, the mask is not constant among the elements in the cp.async instruction.
    # We have weak_mask to allow weak checking that only use the mask of the first element of each cp.async instruction
    # as the mask of the instruction as a whole. there will not be illegal memory access, but it's the user's
    # responsibility to not rely on the data in shared memory that is out-of-bounds in corresponding global memory.

    @staticmethod
    def create(
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
        evict: Optional[str] = None,
        weak_mask: bool = False,
    ) -> CopyAsyncInst:
        if dims is None and len(src.shape) != len(dst.shape):
            raise ValueError(
                f"Source tensor shape {src.shape} and destination tensor shape {dst.shape} must have the same number of dimensions if dims is not provided."
            )
        offsets_ = tuple(as_expr(offset) for offset in offsets)
        return CopyAsyncInst(
            output=None,
            inputs=(dst, src),
            offsets=offsets_,
            dims=tuple(dims) if dims else None,
            evict=evict,
            weak_mask=weak_mask,
        )


@dataclass(frozen=True, eq=False)
class CopyAsyncGenericInst(Instruction):
    ptr: Var
    axes: list[Var]
    offset: Expr
    mask: Optional[Expr]
    evict: Optional[str]
    weak_mask: bool

    @staticmethod
    def create(
        dst: SharedTensor,
        ptr: Var,
        f_offset: Callable[[list[Var]], Expr],
        f_mask: Optional[Callable[[list[Var]], Expr]],
        evict: Optional[str] = None,
        weak_mask: bool = False,
    ) -> CopyAsyncGenericInst:
        axes = index_vars(len(dst.shape))
        offset = f_offset(axes)
        mask = f_mask(axes) if f_mask else None
        return CopyAsyncGenericInst(
            output=None, inputs=(dst,), ptr=ptr, axes=axes, offset=offset, mask=mask, evict=evict, weak_mask=weak_mask
        )


@dataclass(frozen=True, eq=False)
class CopyAsyncCommitGroupInst(Instruction):
    @staticmethod
    def create() -> CopyAsyncCommitGroupInst:
        return CopyAsyncCommitGroupInst(output=None, inputs=())


@dataclass(frozen=True, eq=False)
class CopyAsyncWaitGroupInst(Instruction):
    n: Expr

    @staticmethod
    def create(n: Expr) -> CopyAsyncWaitGroupInst:
        return CopyAsyncWaitGroupInst(output=None, inputs=(), n=n)


@dataclass(frozen=True, eq=False)
class CopyAsyncWaitAllInst(Instruction):
    @staticmethod
    def create() -> CopyAsyncWaitAllInst:
        return CopyAsyncWaitAllInst(output=None, inputs=())
