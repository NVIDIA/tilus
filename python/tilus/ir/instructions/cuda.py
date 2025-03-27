from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from hidet.ir.dtypes import DataType, bf16, f16, f32, i8, i32
from hidet.ir.expr import Expr, Var, as_expr
from tilus.extensions.hidet.ir.expr import index_vars
from tilus.ir.inst import Instruction, InstructionConfig
from tilus.ir.layout import RegisterLayout, column_repeat, column_spatial, repeat, spatial
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedTensor


@dataclass(frozen=True, eq=False)
class MmaDotInst(Instruction):
    warp_spatial: tuple[int, int, int]
    warp_repeat: tuple[int, int, int]
    config: MmaDotConfig

    @staticmethod
    def create(
        a: RegisterTensor,
        b: RegisterTensor,
        c: RegisterTensor,
        warp_spatial: Sequence[int],
        warp_repeat: Sequence[int],
        config: MmaDotConfig,
        output: RegisterTensor,
    ) -> MmaDotInst:
        if len(warp_spatial) == 2:
            warp_spatial_ = (warp_spatial[0], warp_spatial[1], 1)
        elif len(warp_spatial) == 3:
            warp_spatial_ = (warp_spatial[0], warp_spatial[1], warp_spatial[2])
        else:
            raise ValueError("warp_spatial must have length 2 or 3")
        if len(warp_repeat) == 3:
            warp_repeat_ = (warp_repeat[0], warp_repeat[1], warp_repeat[2])
        else:
            raise ValueError("warp_repeat must have length 3")
        return MmaDotInst(
            output=output, inputs=(a, b, c), config=config, warp_spatial=warp_spatial_, warp_repeat=warp_repeat_
        )


@dataclass(frozen=True, eq=False)
class SimtDotInst(Instruction):
    warp_spatial: tuple[int, int, int]
    warp_repeat: tuple[int, int, int]
    thread_spatial: tuple[int, int]
    thread_repeat: tuple[int, int]

    @staticmethod
    def create(
        a: RegisterTensor,
        b: RegisterTensor,
        c: RegisterTensor,
        warp_spatial: tuple[int, int, int],
        warp_repeat: tuple[int, int, int],
        thread_spatial: tuple[int, int],
        thread_repeat: tuple[int, int],
        output: Optional[RegisterTensor] = None,
    ) -> SimtDotInst:
        if output is None:
            output = RegisterTensor.create(c.dtype, c.layout)
        return SimtDotInst(
            output=output,
            inputs=(a, b, c),
            warp_spatial=warp_spatial,
            warp_repeat=warp_repeat,
            thread_spatial=thread_spatial,
            thread_repeat=thread_repeat,
        )


@dataclass(frozen=True, eq=False)
class CopyAsyncInst(Instruction):
    offsets: tuple[Expr, ...]
    dims: Optional[tuple[int, ...]]
    evict: Optional[str]

    @staticmethod
    def create(
        src: GlobalTensor,
        dst: SharedTensor,
        offsets: Sequence[Expr | int],
        dims: Optional[Sequence[int]] = None,
        evict: Optional[str] = None,
    ) -> CopyAsyncInst:
        offsets_ = tuple(as_expr(offset) for offset in offsets)
        return CopyAsyncInst(
            output=None, inputs=(dst, src), offsets=offsets_, dims=tuple(dims) if dims else None, evict=evict
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


@dataclass(frozen=True, eq=False)
class LoadMatrixInst(Instruction):
    ptr: Var
    axes: tuple[Var, ...]
    offset: Expr
    config: LoadMatrixConfig

    @staticmethod
    def create(
        ptr: Var,
        axes: Sequence[Var],
        offset: Expr,
        config: LoadMatrixConfig,
        output: RegisterTensor,
    ) -> LoadMatrixInst:
        assert len(axes) == len(output.shape)

        return LoadMatrixInst(inputs=(), output=output, ptr=ptr, axes=tuple(axes), offset=offset, config=config)


@dataclass(frozen=True, eq=False)
class LockSemaphoreInst(Instruction):
    semaphore: Expr
    value: Expr

    @staticmethod
    def create(
        semaphore: Expr,
        value: Expr,
    ) -> LockSemaphoreInst:
        return LockSemaphoreInst(inputs=(), output=None, semaphore=semaphore, value=value)


@dataclass(frozen=True, eq=False)
class ReleaseSemaphoreInst(Instruction):
    semaphore: Expr
    value: Expr

    @staticmethod
    def create(
        semaphore: Expr,
        value: Expr,
    ) -> ReleaseSemaphoreInst:
        return ReleaseSemaphoreInst(inputs=(), output=None, semaphore=semaphore, value=value)


@dataclass(frozen=True, eq=False)
class LoadMatrixConfig(InstructionConfig):
    nbytes: int
    trans: bool
    ldmatrix_layout: RegisterLayout

    @staticmethod
    @functools.cache
    def all() -> tuple[LoadMatrixConfig, ...]:
        return (
            LoadMatrixConfig(2, False, spatial(8, 4).repeat(1, 2)),
            LoadMatrixConfig(2, True, column_spatial(4, 8).repeat(2, 1)),
        )


@dataclass(frozen=True, eq=False)
class MmaDotConfig(InstructionConfig):
    name: str
    m: int
    n: int
    k: int
    vec_k: int
    la: RegisterLayout
    lb: RegisterLayout
    lc: RegisterLayout
    operand_type: DataType
    acc_type: DataType

    def __hash__(self):
        return hash((MmaDotConfig, self.name))

    def __eq__(self, other):
        return isinstance(other, MmaDotConfig) and self.name == other.name

    def hidet_mma_config(self):
        from hidet.ir.primitives.cuda.mma import MmaConfig

        v_pos = self.name.find("v")
        under_pos = self.name.find("_", v_pos)
        hidet_config_name = self.name[:v_pos] + self.name[under_pos:]

        return getattr(MmaConfig, hidet_config_name)()

    @staticmethod
    @functools.cache
    def m16n8k16_f16_f16(vec_k: int = 1) -> MmaDotConfig:
        return MmaDotConfig(
            name="m16n8k16v{}_f16_f16".format(vec_k),
            m=16,
            n=8,
            k=16,
            vec_k=vec_k,
            la=column_repeat(2, 2).spatial(8, 4).repeat(1, vec_k * 2),
            lb=repeat(2, 1).column_spatial(4, 8).repeat(vec_k * 2, 1),
            lc=repeat(2, 1).spatial(8, 4).repeat(1, 2),
            operand_type=f16,
            acc_type=f16,
        )

    @staticmethod
    @functools.cache
    def m16n8k16_f16_f32(vec_k: int = 1) -> MmaDotConfig:
        return MmaDotConfig(
            name="m16n8k16v{}_f16_f32".format(vec_k),
            m=16,
            n=8,
            k=16,
            vec_k=vec_k,
            la=column_repeat(2, 2).spatial(8, 4).repeat(1, vec_k * 2),
            lb=repeat(2, 1).column_spatial(4, 8).repeat(vec_k * 2, 1),
            lc=repeat(2, 1).spatial(8, 4).repeat(1, 2),
            operand_type=f16,
            acc_type=f32,
        )

    @staticmethod
    @functools.cache
    def m16n8k16_bf16_f32(vec_k: int = 1) -> MmaDotConfig:
        return MmaDotConfig(
            name="m16n8k16v{}_bf16_f32".format(vec_k),
            m=16,
            n=8,
            k=16,
            vec_k=vec_k,
            la=column_repeat(2, 2).spatial(8, 4).repeat(1, vec_k * 2),
            lb=repeat(2, 1).column_spatial(4, 8).repeat(vec_k * 2, 1),
            lc=repeat(2, 1).spatial(8, 4).repeat(1, 2),
            operand_type=bf16,
            acc_type=f32,
        )

    @staticmethod
    @functools.cache
    def m8n8k16_i8_i32(vec_k: int = 1) -> MmaDotConfig:
        return MmaDotConfig(
            name="m8n8k16v{}_i8_i32".format(vec_k),
            m=8,
            n=8,
            k=16,
            vec_k=vec_k,
            la=spatial(8, 4).repeat(1, 4),
            lb=column_spatial(4, 8).repeat(4, 1),
            lc=spatial(8, 4).repeat(1, 2),
            operand_type=i8,
            acc_type=i32,
        )

    @staticmethod
    @functools.cache
    def all() -> dict[str, MmaDotConfig]:
        config_list = []
        for vec_k in [1, 2, 3, 4]:
            config_list.append(MmaDotConfig.m16n8k16_f16_f32(vec_k))
            config_list.append(MmaDotConfig.m16n8k16_f16_f16(vec_k))
            config_list.append(MmaDotConfig.m16n8k16_bf16_f32(vec_k))
            config_list.append(MmaDotConfig.m8n8k16_i8_i32(vec_k))
        return {config.name: config for config in config_list}

    @staticmethod
    @functools.cache
    def from_name(name: str) -> MmaDotConfig:
        return MmaDotConfig.all()[name]
