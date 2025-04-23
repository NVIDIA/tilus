from __future__ import annotations

import functools
from dataclasses import dataclass

from hidet.ir.dtypes import bf16, f16, f32, i8, i32, uint32
from hidet.ir.expr import Expr, cast, if_then_else, var
from hidet.ir.type import DataType
from tilus.backends.codegen import BaseInstEmitter, register_emitter
from tilus.extensions.hidet.ir.primitives.cuda.mma import mma_sync_v2
from tilus.ir.instructions.cuda import MmaDotInst
from tilus.ir.layout import RegisterLayout, column_repeat, column_spatial, repeat, spatial
from tilus.ir.tensor import RegisterTensor
from tilus.target import nvgpu_sm70


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
            la=column_repeat(2, 2).spatial(8, 4).repeat(1, vec_k * 2),
            lb=repeat(2, 1).column_spatial(4, 8).repeat(vec_k * 2, 1),
            lc=repeat(2, 1).spatial(8, 4).repeat(1, 2),
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
            la=column_repeat(2, 2).spatial(8, 4).repeat(1, vec_k * 2),
            lb=repeat(2, 1).column_spatial(4, 8).repeat(vec_k * 2, 1),
            lc=repeat(2, 1).spatial(8, 4).repeat(1, 2),
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
            la=column_repeat(2, 2).spatial(8, 4).repeat(1, vec_k * 2),
            lb=repeat(2, 1).column_spatial(4, 8).repeat(vec_k * 2, 1),
            lc=repeat(2, 1).spatial(8, 4).repeat(1, 2),
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
            la=spatial(8, 4).repeat(1, 4 * vec_k),
            lb=column_spatial(4, 8).repeat(4 * vec_k, 1),
            lc=spatial(8, 4).repeat(1, 2),
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
            la=column_repeat(2, 1).spatial(8, 4).repeat(1, vec_k * 4),
            lb=column_spatial(4, 8).repeat(vec_k * 4, 1),
            lc=repeat(2, 1).spatial(8, 4).repeat(1, 2),
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
            la=column_repeat(2, 2).spatial(8, 4).repeat(1, vec_k * 4),
            lb=repeat(2, 1).column_spatial(4, 8).repeat(vec_k * 4, 1),
            lc=repeat(2, 1).spatial(8, 4).repeat(1, 2),
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


@register_emitter(MmaDotInst, target=nvgpu_sm70)
class MmaDotInstEmitter(BaseInstEmitter):
    @staticmethod
    def resolve_mma_config(
        a: RegisterTensor, b: RegisterTensor, c: RegisterTensor, d: RegisterTensor
    ) -> tuple[AtomicMmaConfig, tuple[RegisterLayout, ...]]:
        if a.dtype != b.dtype:
            raise ValueError("a and b must have the same dtype, got {} and {}".format(a.dtype, b.dtype))
        if c.dtype != d.dtype:
            raise ValueError("c and d must have the same dtype, got {} and {}".format(c.dtype, d.dtype))
        for config in AtomicMmaConfig.all_configs().values():
            if a.dtype != config.operand_type or c.dtype != config.acc_type:
                # dtype mismatch, skip
                continue
            outers = [
                p / q
                for p, q in zip([a.layout, b.layout, c.layout, d.layout], [config.la, config.lb, config.lc, config.lc])
            ]
            if any(outer is None for outer in outers):
                # layout not divisible, skip
                continue
            outer_a, outer_b, outer_c, outer_d = outers
            assert len(outer_a.shape) == len(outer_b.shape) == len(outer_c.shape) == len(outer_d.shape) == 2
            assert outer_a.shape[0] == outer_c.shape[0] == outer_d.shape[0]  # m
            assert outer_a.shape[1] == outer_b.shape[0]  # k
            assert outer_b.shape[1] == outer_c.shape[1] == outer_d.shape[1]  # n
            m_size, n_size, k_size = outer_a.shape[0], outer_c.shape[1], outer_a.shape[1]

            a_workers: dict[tuple[int, int], list[int]] = outer_a.get_worker_map()
            b_workers: dict[tuple[int, int], list[int]] = outer_b.get_worker_map()
            c_workers: dict[tuple[int, int], list[int]] = outer_c.get_worker_map()
            d_workers: dict[tuple[int, int], list[int]] = outer_d.get_worker_map()

            for i in range(m_size):
                for j in range(n_size):
                    assert len(c_workers[(i, j)]) == len(d_workers[(i, j)]) == 1
                    assert c_workers[(i, j)][0] == d_workers[(i, j)][0]
                    acc_worker = c_workers[(i, j)][0]
                    for k in range(k_size):
                        assert (i, k) in a_workers and acc_worker in a_workers[(i, k)]
                        assert (k, j) in b_workers and acc_worker in b_workers[(k, j)]
            return config, (outer_a, outer_b, outer_c, outer_d)
        msg = (
            "Can not resolve the mma config for\n"
            + " a: {}{} {}\n".format(a.dtype.name, list(a.shape), a.layout)
            + " b: {}{} {}\n".format(b.dtype.name, list(b.shape), b.layout)
            + " c: {}{} {}\n".format(c.dtype.name, list(c.shape), c.layout)
            + " d: {}{} {}\n".format(d.dtype.name, list(d.shape), d.layout)
        )
        raise RuntimeError(msg)

    def emit(self, inst: MmaDotInst) -> None:  # type: ignore
        a_tensor = inst.inputs[0].as_register_tensor()
        b_tensor = inst.inputs[1].as_register_tensor()
        c_tensor = inst.inputs[2].as_register_tensor()
        d_tensor = inst.register_output
        config, outers = self.resolve_mma_config(a_tensor, b_tensor, c_tensor, d_tensor)
        a_buf = self.tensor2var[a_tensor]
        b_buf = self.tensor2var[b_tensor]
        c_buf = self.tensor2var[c_tensor]
        d_buf = self.get_or_allocate_var(tensor=d_tensor, name="d")
        outer_a, outer_b, outer_c, outer_d = outers
        k_size = outer_a.shape[1]

        warp_id: Expr = self.current_worker // 32
        with self.for_range(outer_c.local_size) as c_outer_local:
            c_outer_indices = outer_c.local2global(c_outer_local, worker=warp_id)
            d_outer_indices = c_outer_indices
            c_local = c_outer_local * config.lc.local_size
            d_local = outer_d.global2local(d_outer_indices, worker=warp_id) * config.lc.local_size
            with self.for_range(k_size) as k:
                i, j = c_outer_indices
                a_outer_indices = (i, k)
                b_outer_indices = (k, j)
                a_local = outer_a.global2local(a_outer_indices, worker=warp_id) * config.la.local_size
                b_local = outer_b.global2local(b_outer_indices, worker=warp_id) * config.lb.local_size

                with self.for_range(config.vec_k) as k_inner:
                    a_regs = self.declare(var("a_regs", ~uint32), init=cast(~a_buf[a_local], ~uint32))
                    b_regs = self.declare(var("b_regs", ~uint32), init=cast(~b_buf[b_local], ~uint32))
                    if c_tensor is d_tensor:
                        c_regs = self.declare(var("c_regs", ~uint32), init=cast(~c_buf[c_local], ~uint32))
                    else:
                        c_regs = self.declare(
                            var("c_regs", ~uint32),
                            init=cast(if_then_else(k == 0, ~c_buf[c_local], ~d_buf[d_local]), ~uint32),
                        )
                    d_regs = self.declare(var("d_regs", ~uint32), init=cast(~d_buf[d_local], ~uint32))

                    self.append(
                        mma_sync_v2(
                            config=config.hidet_mma_config(),
                            d_reg_p=[d_regs + i for i in range(config.hidet_mma_config().c_regs)],
                            a_reg_p=[
                                a_regs + i * config.vec_k + k_inner for i in range(config.hidet_mma_config().a_regs)
                            ],
                            b_reg_p=[
                                b_regs + i * config.vec_k + k_inner for i in range(config.hidet_mma_config().b_regs)
                            ],
                            c_reg_p=[c_regs + i for i in range(config.hidet_mma_config().c_regs)],
                        )
                    )

        # warp_spatial: Tuple[int, int, int] = inst.warp_spatial
        # warp_repeat: Tuple[int, int, int] = inst.warp_repeat
        # c_outer_shape = c_tensor.shape[:-2]
        #
        # with self.for_grid(c_outer_shape) as c_outer_indices:  # type: ignore
        #     a_outer_indices = broadcast_indices(c_outer_indices, a_tensor.shape[:-2], c_outer_shape)
        #     b_outer_indices = broadcast_indices(c_outer_indices, b_tensor.shape[:-2], c_outer_shape)
        #     with self.for_grid(list(warp_repeat)) as repeat_indices:
        #         from hidet.ir.mapping import spatial_map
        #
        #         spatial_indices: Tuple[Expr, Expr, Expr] = spatial_map(warp_spatial, ranks=[1, 2, 0])(warp_id)[0]
        #
        #         mma_indices = [
        #             (spatial_indices[0] * warp_repeat[0] + repeat_indices[0]) * config.m,
        #             (spatial_indices[1] * warp_repeat[1] + repeat_indices[1]) * config.n,
        #             (spatial_indices[2] * warp_repeat[2] + repeat_indices[2]) * (config.k * config.vec_k),
        #         ]
        #
        #         a_indices = a_outer_indices + [mma_indices[0], mma_indices[2]]
        #         b_indices = b_outer_indices + [mma_indices[2], mma_indices[1]]
        #         c_indices = c_outer_indices + [mma_indices[0], mma_indices[1]]
        #         d_indices = c_indices
        #
        #         a_regs = self.declare(
        #             var("a_regs", ~uint32),
        #             init=cast(~a_buf[a_tensor.layout.global2local(a_indices, worker=self.current_worker)], ~uint32),
        #         )
        #         b_regs = self.declare(
        #             var("b_regs", ~uint32),
        #             init=cast(~b_buf[b_tensor.layout.global2local(b_indices, worker=self.current_worker)], ~uint32),
        #         )
        #         c_regs = self.declare(
        #             var("c_regs", ~uint32),
        #             init=if_then_else(  # we reduce over the warp_repeat[2] dimension locally
        #                 repeat_indices[2] == 0,
        #                 cast(~c_buf[c_tensor.layout.global2local(c_indices, worker=self.current_worker)], ~uint32),
        #                 cast(~d_buf[d_tensor.layout.global2local(d_indices, worker=self.current_worker)], ~uint32),
        #             ),
        #         )
        #         d_regs = self.declare(
        #             var("d_regs", ~uint32),
        #             init=cast(~d_buf[d_tensor.layout.global2local(d_indices, worker=self.current_worker)], ~uint32),
        #         )
        #
        #         with self.for_range(config.vec_k) as vk:
        #             hidet_mma: HidetMmaConfig = config.hidet_mma_config()
        #             self.append(
        #                 mma_sync_v2(
        #                     config=hidet_mma,
        #                     d_reg_p=[d_regs + i for i in range(hidet_mma.c_regs)],
        #                     a_reg_p=[a_regs + i * config.vec_k + vk for i in range(hidet_mma.a_regs)],
        #                     b_reg_p=[b_regs + i * config.vec_k + vk for i in range(hidet_mma.b_regs)],
        #                     c_reg_p=[c_regs + i for i in range(hidet_mma.c_regs)],
        #                 )
        #             )
