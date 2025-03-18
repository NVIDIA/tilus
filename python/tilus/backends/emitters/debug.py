from typing import List, Sequence

from hidet.ir.dtypes import int4b, uint8, int8, uint32, int32, float16, float32, bfloat16
from hidet.ir.dtypes import uint4b
from hidet.ir.expr import Expr, logical_and, cast
from hidet.ir.primitives.debug import printf

from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.extensions.hidet.ir.dtypes import float8_e4m3, float6_e3m2
from tilus.ir.inst import PrintValueInst, FormatPrintInst
from tilus.ir.layout import Layout
from tilus.ir.value import RegisterValue, SharedValue, SharedLayout
from tilus.target import gpgpu_any
from tilus.utils import prod


@register_inst_emitter(PrintValueInst, target=gpgpu_any)
class PrintValueInstEmitter(BaseInstEmitter):
    def print_left_bracket(self, indices: List[Expr], shape: List[int]):
        # left [
        if len(shape) >= 1:
            with self.if_then(logical_and(self.current_worker == 0, indices[-1] == 0)):
                for dim in range(len(indices)):
                    left_cond = logical_and(*[axis == 0 for axis in indices[dim:]])
                    with self.if_then(left_cond):
                        self.append(printf("["))
                    with self.otherwise():
                        self.append(printf(" "))
            self.sync()

    def print_right_bracket(self, indices: Sequence[Expr], shape: Sequence[int]):
        # right ]
        if len(shape) >= 1:
            with self.if_then(logical_and(self.current_worker == 0, indices[-1] == shape[-1] - 1)):
                for dim in reversed(range(len(indices))):
                    right_cond = logical_and(*[axis == extent - 1 for axis, extent in zip(indices[dim:], shape[dim:])])
                    with self.if_then(right_cond):
                        self.append(printf("]"))
                self.append(printf("\n"))
            self.sync()

    def print_seperate_comma(self, indices: Sequence[Expr], shape: Sequence[int]):
        if len(shape) >= 1:
            with self.if_then(logical_and(self.current_worker == 0, indices[-1] != shape[-1] - 1)):
                self.append(printf(", "))
            self.sync()

    def restore_indices(self, squeezed_indices: Sequence[Expr], squeezed_dims: Sequence[int], shape: Sequence[int]):
        indices: List[Expr] = []
        for dim in range(len(shape)):
            if dim in squeezed_dims:
                indices.append(squeezed_indices[squeezed_dims.index(dim)])
            else:
                indices.append(int32(0))
        return indices

    def emit(self, inst: PrintValueInst):
        default_fmt_mapping = {
            int4b: "%2d",
            uint4b: "%2d",
            uint8: "%3d",
            int8: "%3d",
            int32: "%5d",
            float8_e4m3: "%5.2f",
            float6_e3m2: "%5.2f",
            bfloat16: "%5.3f",
            float16: "%5.2f",
            float32: "%6.3f",
            uint32: "%3u",
        }
        value = inst.inputs[0]
        dtype = value.dtype
        shape: Sequence[int] = value.shape
        squeezed_dims = [dim for dim in range(len(shape)) if shape[dim] > 1]
        squeezed_shape = [shape[dim] for dim in squeezed_dims]
        not_supported_print = inst.inputs[0].dtype.is_vector()
        cond = inst.cond

        if isinstance(value, RegisterValue):
            if self.thread_groups.group_size[-1] != value.layout.num_workers:
                # msg = (
                #     'Trying to print a register value with layout: \n{}\nin a thread group with group size: {}'.format(
                #         value.layout, self.thread_groups.group_size[-1]
                #     )
                # )
                # raise ValueError(msg)
                pass

            layout: Layout = value.layout
            self.sync()
            with self.if_then(cond):
                self.sync()
                with self.if_then(self.current_worker == 0):
                    self.append(
                        printf(
                            "%s%s\n",
                            inst.msg,
                            "register_tile(dtype={}, shape={}) layout={}".format(
                                value.dtype.name, value.shape, value.layout
                            ),
                        )
                    )
                self.sync()
                if not not_supported_print:
                    fmt = inst.fmt if inst.fmt is not None else default_fmt_mapping[value.dtype]
                    with self.for_grid(squeezed_shape) as squeezed_indices:
                        self.print_left_bracket(squeezed_indices, squeezed_shape)

                        if prod(layout.shape) != layout.local_size * layout.num_workers:
                            with self.if_then(self.current_worker == 0):
                                self.append(printf("{"))
                        self.sync()

                        # print the element
                        indices = self.restore_indices(squeezed_indices, squeezed_dims, shape)
                        is_valid = layout.is_valid(global_indices=indices, worker=self.current_worker)
                        with self.if_then(logical_and(self.current_worker < layout.num_workers, is_valid)):
                            buf = self.value2var[value]
                            local_index = layout.global2local(indices, worker=self.current_worker)
                            data = buf[local_index]
                            if dtype.is_float():
                                data = cast(data, float32)
                            elif dtype.is_integer():
                                data = cast(data, int32)
                            else:
                                raise NotImplementedError()

                            if prod(layout.shape) == layout.local_size * layout.num_workers:
                                self.append(printf(fmt, data))
                            else:
                                # multi threads store the same value
                                self.append(printf("%3d:" + fmt + " ", self.current_worker, data))
                        self.sync()

                        if prod(layout.shape) != layout.local_size * layout.num_workers:
                            with self.if_then(self.current_worker == 0):
                                self.append(printf("}"))
                        self.sync()

                        self.print_seperate_comma(squeezed_indices, squeezed_shape)

                        self.print_right_bracket(squeezed_indices, squeezed_shape)
        elif isinstance(value, SharedValue):
            shared_layout: SharedLayout = value.layout
            buf = self.value2var[value]
            self.sync()
            with self.if_then(cond):
                self.sync()
                with self.if_then(self.current_worker == 0):
                    self.append(printf(inst.msg))
                    self.append(
                        printf(
                            "shared_tile(dtype={}, shape={}) layout={}\n".format(
                                value.dtype.name, value.shape, value.layout
                            )
                        )
                    )
                self.sync()
                if not not_supported_print:
                    fmt = inst.fmt if inst.fmt is not None else default_fmt_mapping[value.dtype]
                    with self.for_grid(squeezed_shape) as squeezed_indices:
                        self.print_left_bracket(squeezed_indices, squeezed_shape)

                        data_indices = self.restore_indices(squeezed_indices, squeezed_dims, shape)
                        offset = shared_layout(*data_indices)
                        data = buf[offset]
                        if dtype.is_float():
                            data = cast(data, float32)
                        elif dtype.is_integer():
                            data = cast(data, int32)
                        else:
                            raise NotImplementedError()
                        with self.if_then(self.current_worker == 0):
                            self.append(printf(fmt, data))
                        self.sync()

                        self.print_seperate_comma(squeezed_indices, squeezed_shape)
                        self.print_right_bracket(squeezed_indices, squeezed_shape)
            self.sync()
        else:
            raise NotImplementedError()


@register_inst_emitter(FormatPrintInst, target=gpgpu_any)
class FormatPrintInstEmitter(BaseInstEmitter):
    def emit(self, inst: FormatPrintInst):
        self.sync()
        with self.if_then(logical_and(inst.cond, self.current_worker == 0)):
            self.append(printf(inst.fstring, *inst.expressions))
        self.sync()
