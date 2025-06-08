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
    check_bounds: bool = True

    @staticmethod
    def create(
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
        evict: Optional[str] = None,
        check_bounds: bool = True,
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
            check_bounds=check_bounds,
        )


@dataclass(frozen=True, eq=False)
class CopyAsyncGenericInst(Instruction):
    ptr: Var
    axes: list[Var]
    offset: Expr
    mask: Optional[Expr]
    evict: Optional[str]

    @staticmethod
    def create(
        dst: SharedTensor,
        ptr: Var,
        f_offset: Callable[[list[Var]], Expr],
        f_mask: Optional[Callable[[list[Var]], Expr]],
        evict: Optional[str] = None,
    ) -> CopyAsyncGenericInst:
        axes = index_vars(len(dst.shape))
        offset = f_offset(axes)
        mask = f_mask(axes) if f_mask else None
        return CopyAsyncGenericInst(
            output=None, inputs=(dst,), ptr=ptr, axes=axes, offset=offset, mask=mask, evict=evict
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
