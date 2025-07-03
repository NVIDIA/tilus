from __future__ import annotations

import functools
from dataclasses import dataclass

from hidet.ir.dtypes import bf16, f16, f32, i8, i32
from hidet.ir.type import DataType

from tilus.ir.inst import Instruction
from tilus.ir.layout import RegisterLayout, column_local, column_spatial, local, spatial
from tilus.ir.tensor import RegisterTensor


@dataclass(frozen=True, eq=False)
class DotInst(Instruction):
    @staticmethod
    def create(
        a: RegisterTensor,
        b: RegisterTensor,
        c: RegisterTensor,
        output: RegisterTensor,
    ) -> DotInst:
        return DotInst(
            output=output,
            inputs=(a, b, c),
        )


@dataclass(frozen=True, eq=False)
class AtomicMmaConfig:
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
        return hash((AtomicMmaConfig, self.name))

    def __eq__(self, other):
        return isinstance(other, AtomicMmaConfig) and self.name == other.name

    def hidet_mma_config(self):
        from hidet.ir.primitives.cuda.mma import MmaConfig

        v_pos = self.name.find("v")
        under_pos = self.name.find("_", v_pos)
        hidet_config_name = self.name[:v_pos] + self.name[under_pos:]

        return getattr(MmaConfig, hidet_config_name)()

    @staticmethod
    @functools.cache
    def m16n8k16_f16_f16(vec_k: int = 1) -> AtomicMmaConfig:
        return AtomicMmaConfig(
            name="m16n8k16v{}_f16_f16".format(vec_k),
            m=16,
            n=8,
            k=16,
            vec_k=vec_k,
            la=column_local(2, 2).spatial(8, 4).local(1, vec_k * 2),
            lb=local(2, 1).column_spatial(4, 8).local(vec_k * 2, 1),
            lc=local(2, 1).spatial(8, 4).local(1, 2),
            operand_type=f16,
            acc_type=f16,
        )

    @staticmethod
    @functools.cache
    def m16n8k16_f16_f32(vec_k: int = 1) -> AtomicMmaConfig:
        return AtomicMmaConfig(
            name="m16n8k16v{}_f16_f32".format(vec_k),
            m=16,
            n=8,
            k=16,
            vec_k=vec_k,
            la=column_local(2, 2).spatial(8, 4).local(1, vec_k * 2),
            lb=local(2, 1).column_spatial(4, 8).local(vec_k * 2, 1),
            lc=local(2, 1).spatial(8, 4).local(1, 2),
            operand_type=f16,
            acc_type=f32,
        )

    @staticmethod
    @functools.cache
    def m16n8k16_bf16_f32(vec_k: int = 1) -> AtomicMmaConfig:
        return AtomicMmaConfig(
            name="m16n8k16v{}_bf16_f32".format(vec_k),
            m=16,
            n=8,
            k=16,
            vec_k=vec_k,
            la=column_local(2, 2).spatial(8, 4).local(1, vec_k * 2),
            lb=local(2, 1).column_spatial(4, 8).local(vec_k * 2, 1),
            lc=local(2, 1).spatial(8, 4).local(1, 2),
            operand_type=bf16,
            acc_type=f32,
        )

    @staticmethod
    @functools.cache
    def m8n8k16_i8_i32(vec_k: int = 1) -> AtomicMmaConfig:
        return AtomicMmaConfig(
            name="m8n8k16v{}_i8_i32".format(vec_k),
            m=8,
            n=8,
            k=16,
            vec_k=vec_k,
            la=spatial(8, 4).local(1, 4 * vec_k),
            lb=column_spatial(4, 8).local(4 * vec_k, 1),
            lc=spatial(8, 4).local(1, 2),
            operand_type=i8,
            acc_type=i32,
        )

    @staticmethod
    @functools.cache
    def m16n8k16_i8_i32(vec_k: int = 1) -> AtomicMmaConfig:
        return AtomicMmaConfig(
            name="m16n8k16v{}_i8_i32".format(vec_k),
            m=16,
            n=8,
            k=16,
            vec_k=vec_k,
            la=column_local(2, 1).spatial(8, 4).local(1, vec_k * 4),
            lb=column_spatial(4, 8).local(vec_k * 4, 1),
            lc=local(2, 1).spatial(8, 4).local(1, 2),
            operand_type=i8,
            acc_type=i32,
        )

    @staticmethod
    @functools.cache
    def m16n8k32_i8_i32(vec_k: int = 1) -> AtomicMmaConfig:
        return AtomicMmaConfig(
            name="m16n8k32v{}_i8_i32".format(vec_k),
            m=16,
            n=8,
            k=32,
            vec_k=vec_k,
            la=column_local(2, 2).spatial(8, 4).local(1, vec_k * 4),
            lb=local(2, 1).column_spatial(4, 8).local(vec_k * 4, 1),
            lc=local(2, 1).spatial(8, 4).local(1, 2),
            operand_type=i8,
            acc_type=i32,
        )

    @staticmethod
    @functools.cache
    def all_configs() -> dict[str, AtomicMmaConfig]:
        config_list: list[AtomicMmaConfig] = []
        for vec_k in [1, 2, 3, 4]:
            config_list.append(AtomicMmaConfig.m16n8k16_f16_f32(vec_k))
            config_list.append(AtomicMmaConfig.m16n8k16_f16_f16(vec_k))
            config_list.append(AtomicMmaConfig.m16n8k16_bf16_f32(vec_k))
            config_list.append(AtomicMmaConfig.m8n8k16_i8_i32(vec_k))
            config_list.append(AtomicMmaConfig.m16n8k16_i8_i32(vec_k))
            config_list.append(AtomicMmaConfig.m16n8k32_i8_i32(vec_k))
        return {config.name: config for config in config_list}
