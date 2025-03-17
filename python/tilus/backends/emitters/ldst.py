from typing import List, Optional, Union, Tuple, Dict, Sequence

from hidet.ir.dtypes import uint32, uint16, uint8, int32, boolean, vectorize
from hidet.ir.expr import Expr, if_then_else, cast, Var, logical_and
from hidet.ir.type import DataType, void_p

from tilus.backends.codegen import BaseInstEmitter, register_inst_emitter
from tilus.extensions.hidet.ir.expr import index_vars
from tilus.extensions.hidet.ir.type import type_equal
from tilus.extensions.hidet.ir.utils.index_transform import index_add
from tilus.extensions.hidet.ir.tools.rewriter import rewrite
from tilus.extensions.hidet.ir.primitives.cuda.ldst import load, store
from tilus.ir.analyzers.value_analyzer import analyze_info, TensorInfo
from tilus.ir.inst import LoadGlobalInst, StoreGlobalInst, LoadSharedInst, StoreSharedInst
from tilus.ir.layout import Layout
from tilus.ir.value import RegisterValue
from tilus.ir.value import SharedLayout
from tilus.target import gpgpu_any, nvgpu_any, amdgpu_any
from tilus.utils import gcd


class LoadStoreInstBaseEmitter(BaseInstEmitter):
    def __init__(self, codegen) -> None:
        super().__init__(codegen)
        self.vectorize_dimension: Optional[int] = None
        self.vector_bytes: Optional[int] = None

    def analyze_vectorization(self, inst: Union[LoadGlobalInst, StoreGlobalInst, LoadSharedInst, StoreSharedInst]):
        """
        Analyze the applicable vectorization of the load/store instruction to global or shared memory.

        Give a tile of data to be loaded or stored, we try to find the dimension along which we can load/store multiple
        elements at once. We can split the tile of elements into a tile of vectors of elements. And the elements in each
        vector must satisfy:
        1. The elements in each vector must be hold by a single thread.
        2. The vector elements are contiguous in memory.
        3. The vector elements are contiguous in the local storage of the thread.
        4. The mask of loading/storing the vector elements must be identical for all elements in the vector.

        We will determine the dimension along which we can vectorize the load/store, and the number of bytes in each
        vector. We support all normal data types (like 8-bit, 16-bit, or 32-bit data) and all sub-byte data types (like
        1-bit, 2-bit, ... and 7-bit data type).
        """
        # get the register value that is going to be stored or loaded to.
        value: RegisterValue
        if isinstance(inst, (LoadGlobalInst, LoadSharedInst)):
            value = inst.register_output
        elif isinstance(inst, StoreGlobalInst):
            value = inst.inputs[0].as_register_value()
        elif isinstance(inst, StoreSharedInst):
            value = inst.inputs[1].as_register_value()
        else:
            raise NotImplementedError()

        num_dims = len(value.shape)
        dtype: DataType = value.dtype

        var2info = {}
        for var, divisibility in self.codegen.program.var2divisibility.items():
            var2info[var] = TensorInfo.from_divisiblity(shape=value.shape, divisibility=divisibility)

        # analyze the offset and mask's value information (e.g., divisibility, constancy, etc.)
        if isinstance(inst, (LoadGlobalInst, StoreGlobalInst)):
            offset_info: TensorInfo = analyze_info(shape=value.shape, axes=inst.axes, var2info={}, expr=inst.offset)
            mask_info: TensorInfo = analyze_info(
                shape=value.shape,
                axes=inst.axes,
                var2info=var2info,
                expr=inst.mask if inst.mask is not None else boolean.true,
            )
        elif isinstance(inst, (LoadSharedInst, StoreSharedInst)):
            shared_layout: SharedLayout = inst.inputs[0].as_shared_value().layout
            offset_info = analyze_info(
                shape=value.shape, axes=shared_layout.axes, var2info=var2info, expr=shared_layout.offset
            )
            mask_info = analyze_info(
                shape=value.shape, axes=index_vars(len(value.shape)), var2info=var2info, expr=boolean.true
            )
        else:
            raise NotImplementedError()

        # analyze the layout of the tile so that we can know how the elements are distributed stored in threads
        layout = value.layout
        axes = index_vars(len(layout.shape) + 1)
        expr = layout.global2local(global_indices=axes[:-1], worker=axes[-1])
        layout_info: TensorInfo = analyze_info(shape=layout.shape, axes=axes[:-1], var2info={}, expr=expr)

        # enumerate each dimension and check whether we can vectorize on that dimension
        for i in range(num_dims):
            max_vector_elements = gcd(  # to be eligible for vectorized loading, the elements must:
                offset_info[i].continuity,  # contiguous in global/shared memory (cond. 2)
                layout_info[i].continuity,  # contiguous in the local storage of the thread (cond. 1 and 3)
                mask_info[i].constancy,  # the mask must be the same for all elements in the vector (cond. 4)
                layout.local_size,  # the local storage must be able to be divided into multiple such vectors
            )
            if max_vector_elements > 1:
                if max_vector_elements * dtype.nbits % 8 != 0:
                    # the vector elements must be able to be represented by multiple bytes
                    continue
                self.vectorize_dimension = i
                self.vector_bytes = max_vector_elements * dtype.nbits // 8
                break
        else:
            # failed to use vectorized loading
            self.vectorize_dimension = None
            self.vector_bytes = None


class LoadInstBaseEmitter(LoadStoreInstBaseEmitter):
    def get_buffer_and_mask(self, inst: Union[LoadGlobalInst, LoadSharedInst], indices) -> Tuple[Expr, Expr]:
        """
        Get the buffer to load from and the mask indicating whether we should perform the loading.

        Parameters
        ----------
        inst: Union[LoadGlobalInst, LoadSharedInst]
            The load instruction.

        indices: List[Expr]
            The indices in the output tile to load the elements from the memory.
        """
        raise NotImplementedError()

    def vectorized_load(self, load_dtype: DataType, buffer: Expr, dst_buffer: Expr, sub_vec_size: int, sub_vec_i: Expr):
        """
        Perform the vectorized loading operation that loads sub_vec_size * len(dst_addrs) elements from the buffer to
        the destination addresses (must be in register scope).

        Parameters
        ----------
        load_dtype: DataType
            The data type of the elements to load.
        buffer: Expr
            The memory buffer to load from. Must be in global scope or shared scope.
        sub_vec_size: int
            The number of elements to load in each sub-vector.
        sub_vec_i: Expr
            The index of the sub-vector to load.
        dst_addrs: List[Expr]
            The destination addresses to store the loaded elements. Must be in register scope.

        Returns
        -------
        call: Expr
            The call expression that performs the vectorized loading operation.
        """
        raise NotImplementedError()

    def load(self, inst, indices: List[Expr]) -> Tuple[Expr, Expr]:
        """
        Load the elements from the memory with the given indices in the output tile.

        Parameters
        ----------
        indices: List[Expr]
            The indices in the output tile to load the elements from the memory.

        Returns
        -------
        loaded_value, mask: Tuple[Expr, Expr]
            The loaded value and the mask indicating whether the loaded value is valid.
        """
        raise NotImplementedError()

    def emit(self, inst: Union[LoadGlobalInst, LoadSharedInst]):
        # create the output var to store the loaded elements
        output: RegisterValue = inst.register_output
        dtype: DataType = output.dtype
        layout: Layout = output.layout
        var = self.get_or_allocate_var(value=output, name="loaded")

        assert layout.num_workers <= self.num_warps * 32

        if layout.num_workers == self.num_warps * 32:
            condition = boolean.true
        else:
            condition = self.current_worker < layout.num_workers

        with self.if_then(condition):
            self.analyze_vectorization(inst)
            if self.vectorize_dimension is not None:
                assert self.vector_bytes is not None
                # vectorized loading.
                total_nbytes = layout.local_size * dtype.nbits // 8
                with self.for_range(extent=total_nbytes // self.vector_bytes) as vec_i:
                    # get the start indices in the tile
                    start_i = vec_i * self.vector_bytes * 8 // dtype.nbits
                    global_indices: List[Expr] = layout.local2global(local_index=start_i, worker=self.current_worker)

                    # get the vector buffer to load from, and the mask
                    buffer, mask = self.get_buffer_and_mask(inst, global_indices)

                    for load_dtype in [uint32, uint16, uint8]:
                        if self.vector_bytes % load_dtype.nbytes == 0:
                            num_units: int = self.vector_bytes // load_dtype.nbytes
                            regs_name: str = "regs_{}".format(load_dtype.short_name)
                            regs_ptr = self.declare_var(regs_name, ~load_dtype, init=cast(~var[start_i], ~load_dtype))
                            sub_vec_size = gcd(num_units, 4)

                            with self.if_then(mask):
                                with self.for_range(extent=num_units // sub_vec_size) as sub_vec_i:
                                    self.append(
                                        self.vectorized_load(
                                            load_dtype=load_dtype,
                                            buffer=buffer,
                                            dst_buffer=regs_ptr,
                                            sub_vec_size=sub_vec_size,
                                            sub_vec_i=sub_vec_i,
                                        )
                                    )
                            with self.otherwise():
                                with self.for_range(extent=num_units) as uint_i:
                                    self.buffer_store(regs_ptr, indices=[uint_i], value=load_dtype.zero)
                            break
            else:
                # we can not load with vectorization, fall back to element-wise loading
                with self.for_range(extent=output.size) as i:
                    global_indices = layout.local2global(local_index=i, worker=self.current_worker)
                    loaded_value, mask = self.load(inst, global_indices)
                    if mask is not None:
                        loaded_value = if_then_else(mask, loaded_value, output.dtype.zero)
                    self.buffer_store(buf=var, indices=[i], value=loaded_value)


class StoreInstBaseEmitter(LoadStoreInstBaseEmitter):
    def get_buffer_and_mask(self, inst: Union[StoreGlobalInst, StoreSharedInst], indices) -> Tuple[Expr, Expr]:
        """
        Get the buffer to load from and the mask indicating whether we should perform the loading.

        Parameters
        ----------
        inst: Union[LoadGlobalInst, LoadSharedInst]
            The load instruction.

        indices: List[Expr]
            The indices in the output tile to load the elements from the memory.
        """
        raise NotImplementedError()

    def vectorized_store(
        self, store_dtype: DataType, buffer: Expr, source_buffer: Expr, sub_vec_size: int, sub_vec_i: Expr
    ):
        """
        Perform the vectorized storing operation that stores sub_vec_size * len(src_addrs) elements from the source
        addresses (must be in register scope) to the buffer.

        Parameters
        ----------
        store_dtype: DataType
            The data type of the elements to store.
        buffer: Expr
            The memory buffer to store to. Must be in global scope or shared scope.
        source_buffer: Expr
            The source address in the register scope to the elements to be stored.
        sub_vec_size: int
            The number of elements to load in each sub-vector.
        sub_vec_i: Expr
            The index of the sub-vector to load.

        Returns
        -------
        call: Expr, optional
            The call expression that performs the vectorized loading operation.
        """
        raise NotImplementedError()

    def store(self, inst, indices: List[Expr], value: Expr, is_first_occurrence: Expr):
        """
        Store the elements to the memory with the given indices in the input tile.

        Parameters
        ----------
        inst: StoreGlobalInst or StoreSharedInst
            The store instruction.

        indices: List[Expr]
            The indices in the input tile to store the elements to the memory.

        value: Tuple[Expr, Expr]
            The value to store to.

        is_first_occurrence: Expr
            The mask indicating whether the current thread is the first thread to store the value to the memory.
        """
        raise NotImplementedError()

    def emit(self, inst: Union[StoreSharedInst, StoreGlobalInst]):
        # get the register value and its corresponding lowered variable to write
        value: RegisterValue
        if isinstance(inst, StoreSharedInst):
            value = inst.inputs[1].as_register_value()
        elif isinstance(inst, StoreGlobalInst):
            value = inst.inputs[0].as_register_value()
        else:
            raise NotImplementedError()
        layout: Layout = value.layout
        var: Var = self.value2var[value]

        assert layout.num_workers <= self.num_warps * 32

        if layout.num_workers == self.num_warps * 32:
            condition = boolean.true
        else:
            condition = self.current_worker < layout.num_workers

        with self.if_then(condition):
            # check if we can use vectorized store
            self.analyze_vectorization(inst)
            if self.vectorize_dimension is not None:
                assert self.vectorize_dimension is not None and self.vector_bytes is not None
                # vectorized store
                total_nbytes = layout.local_size * value.dtype.nbits // 8
                assert total_nbytes % self.vector_bytes == 0
                with self.for_range(extent=total_nbytes // self.vector_bytes) as vec_i:
                    # get the start indices in the tile
                    start_i = vec_i * self.vector_bytes * 8 // value.dtype.nbits
                    global_indices: List[Expr] = layout.local2global(local_index=start_i, worker=self.current_worker)

                    # get the buffer to store to, and the mask indicating whether we should perform the storing
                    buffer, mask = self.get_buffer_and_mask(inst, global_indices)
                    is_first_occurrence = value.layout.is_first_occurrence(
                        local_index=start_i, worker=self.current_worker
                    )
                    mask = logical_and(mask, is_first_occurrence)

                    for store_dtype in [uint32, uint16, uint8]:
                        if self.vector_bytes % store_dtype.nbytes == 0:
                            num_units: int = self.vector_bytes // store_dtype.nbytes
                            regs_name: str = "regs_{}".format(store_dtype.short_name)
                            regs_ptr = self.declare_var(regs_name, ~store_dtype, init=cast(~var[start_i], ~store_dtype))
                            sub_vec_size = gcd(num_units, 4)

                            with self.if_then(mask):
                                with self.for_range(extent=num_units // sub_vec_size) as sub_vec_i:
                                    self.append(
                                        self.vectorized_store(
                                            store_dtype=store_dtype,
                                            buffer=buffer,
                                            source_buffer=regs_ptr,
                                            sub_vec_size=sub_vec_size,
                                            sub_vec_i=sub_vec_i,
                                        )
                                    )
                            break
            else:
                # write the elements one by one
                with self.for_range(extent=value.size) as i:
                    global_indices = layout.local2global(local_index=i, worker=self.current_worker)
                    is_first_occurrence = value.layout.is_first_occurrence(local_index=i, worker=self.current_worker)
                    self.store(inst, indices=global_indices, value=var[i], is_first_occurrence=is_first_occurrence)


@register_inst_emitter(LoadSharedInst, target=nvgpu_any)
class LoadSharedInstNVGPUEmitter(LoadInstBaseEmitter):
    def get_buffer_and_mask(self, inst: LoadSharedInst, indices) -> Tuple[Expr, Expr]:
        src = inst.inputs[0].as_shared_value()
        global_indices: List[Expr] = index_add(inst.offsets, indices)
        remap = {axis: global_index for axis, global_index in zip(src.layout.axes, global_indices)}
        offset: Expr = rewrite(node=src.layout.offset, rewrite_map=remap)
        mask: Expr = boolean.true
        # buf: Var = self.value2var[src]
        # buffer_smem_addr = self.declare_var('smem_addr', int32, cvta_generic_to_shared(generic_addr=~buf[offset]))
        buffer_smem_addr = self.declare_var(
            name="smem_addr", tp=int32, init=self.shared_value_shared_space_addr[src] + offset * src.dtype.nbytes
        )
        return buffer_smem_addr, mask

    def vectorized_load(self, load_dtype: DataType, buffer: Expr, dst_buffer: Expr, sub_vec_size: int, sub_vec_i: Expr):
        # we need to multiply the addr by the size of the data type since the buffer has int32 data type representing
        # an address in shared memory window
        return load(
            dtype=load_dtype,
            addr=buffer + sub_vec_i * sub_vec_size * load_dtype.nbytes,
            dst_addrs=[dst_buffer + sub_vec_i * sub_vec_size + i for i in range(sub_vec_size)],
            space="shared",
        )

    def load(self, inst: LoadSharedInst, indices: List[Expr]):
        src = inst.inputs[0].as_shared_value()
        indices = index_add(inst.offsets, indices)
        remap = {axis: global_index for axis, global_index in zip(src.layout.axes, indices)}
        offset: Expr = rewrite(node=src.layout.offset, rewrite_map=remap)
        buf: Var = self.value2var[src]
        return buf[offset], boolean.true


@register_inst_emitter(StoreSharedInst, target=nvgpu_any)
class StoreSharedInstEmitter(StoreInstBaseEmitter):
    def get_buffer_and_mask(self, inst: StoreSharedInst, indices: Sequence[Expr]) -> Tuple[Expr, Expr]:
        dst = inst.inputs[0].as_shared_value()
        indices_with_offsets: List[Expr] = index_add(inst.offsets, indices)
        remap = {axis: index for axis, index in zip(dst.layout.axes, indices_with_offsets)}
        offset: Expr = rewrite(node=dst.layout.offset, rewrite_map=remap)
        mask: Expr = boolean.true
        # buf: Var = self.value2var[dst]
        # buffer_smem_addr = self.declare_var('smem_addr', int32, cvta_generic_to_shared(generic_addr=~buf[offset]))
        buffer_smem_addr = self.declare_var(
            name="smem_addr", tp=int32, init=self.shared_value_shared_space_addr[dst] + offset * dst.dtype.nbytes
        )
        return buffer_smem_addr, mask

    def vectorized_store(
        self, store_dtype: DataType, buffer: Expr, source_buffer: Expr, sub_vec_size: int, sub_vec_i: Expr
    ):
        source_buffer = cast(source_buffer, ~store_dtype)
        self.append(
            store(
                dtype=store_dtype,
                addr=buffer + sub_vec_i * sub_vec_size * store_dtype.nbytes,
                src_addrs=[source_buffer + sub_vec_i * sub_vec_size + i for i in range(sub_vec_size)],
                space="shared",
            )
        )

    def store(self, inst, indices: Sequence[Expr], value: Expr, is_first_occurrence: Expr):
        buf: Var = self.value2var[inst.inputs[0]]
        indices = index_add(indices, inst.offsets)
        layout: SharedLayout = inst.inputs[0].as_shared_value().layout
        offset: Expr = layout(*indices)
        with self.if_then(is_first_occurrence):
            self.buffer_store(buf, [offset], value=value)


@register_inst_emitter(LoadGlobalInst, target=gpgpu_any)
class LoadGlobalInstEmitter(LoadInstBaseEmitter):
    def get_buffer_and_mask(self, inst: LoadGlobalInst, indices: List[Expr]) -> Tuple[Expr, Expr]:
        dtype = inst.output.dtype
        remap = {axis: global_index for axis, global_index in zip(inst.axes, indices)}
        offset: Expr = rewrite(node=inst.offset, rewrite_map=remap)
        mask: Expr = rewrite(node=inst.mask, rewrite_map=remap) if inst.mask is not None else boolean.true
        assert inst.ptr.hint is not None
        buffer = self.declare_var(inst.ptr.hint, ~dtype, init=cast(inst.ptr, ~dtype) + offset)
        return buffer, mask

    def vectorized_load(self, load_dtype: DataType, buffer: Expr, dst_buffer: Expr, sub_vec_size: int, sub_vec_i: Expr):
        buffer = cast(buffer, ~vectorize(load_dtype, sub_vec_size))
        dst_buffer = cast(dst_buffer, ~vectorize(load_dtype, sub_vec_size))
        self.buffer_store(dst_buffer, indices=[sub_vec_i], value=buffer[sub_vec_i])

    def load(self, inst: LoadGlobalInst, indices: List[Expr]) -> Tuple[Expr, Expr]:
        remap = {axis: index for axis, index in zip(inst.axes, indices)}
        offset: Expr = rewrite(node=inst.offset, rewrite_map=remap)
        mask: Expr = rewrite(node=inst.mask, rewrite_map=remap) if inst.mask is not None else boolean.true
        buf = cast(inst.ptr, ~inst.output.dtype) if not type_equal(inst.ptr.type, ~inst.output.dtype) else inst.ptr
        loaded_value = buf[offset]
        return loaded_value, mask


@register_inst_emitter(LoadSharedInst, target=amdgpu_any)
class LoadSharedInstAMDGPUEmitter(LoadInstBaseEmitter):
    def get_buffer_and_mask(self, inst: LoadSharedInst, indices) -> Tuple[Expr, Expr]:
        src = inst.inputs[0].as_shared_value()
        global_indices: List[Expr] = index_add(inst.offsets, indices)
        remap = {axis: global_index for axis, global_index in zip(src.layout.axes, global_indices)}
        offset: Expr = rewrite(node=src.layout.offset, rewrite_map=remap)
        mask: Expr = boolean.true
        buf: Var = self.value2var[src]
        return ~buf[offset], mask

    def vectorized_load(self, load_dtype: DataType, buffer: Expr, dst_buffer: Expr, sub_vec_size: int, sub_vec_i: Expr):
        buffer = cast(buffer, ~vectorize(load_dtype, sub_vec_size))
        dst_buffer = cast(dst_buffer, ~vectorize(load_dtype, sub_vec_size))
        self.buffer_store(dst_buffer, indices=[sub_vec_i], value=buffer[sub_vec_i])

    def load(self, inst: LoadSharedInst, indices: List[Expr]):
        src = inst.inputs[0].as_shared_value()
        indices = index_add(inst.offsets, indices)
        remap = {axis: global_index for axis, global_index in zip(src.layout.axes, indices)}
        offset: Expr = rewrite(node=src.layout.offset, rewrite_map=remap)
        buf: Var = self.value2var[src]
        return buf[offset], boolean.true


@register_inst_emitter(StoreGlobalInst, target=gpgpu_any)
class StoreGlobalInstEmitter(StoreInstBaseEmitter):
    def get_buffer_and_mask(self, inst: StoreGlobalInst, indices: List[Expr]) -> Tuple[Expr, Expr]:
        remap = {axis: global_index for axis, global_index in zip(inst.axes, indices)}
        offset: Expr = rewrite(node=inst.offset, rewrite_map=remap)
        mask: Expr = rewrite(node=inst.mask, rewrite_map=remap) if inst.mask is not None else boolean.true
        buf = (
            cast(inst.ptr, ~inst.inputs[0].as_register_value().dtype) if type_equal(inst.ptr.type, void_p) else inst.ptr
        )
        start_ptr = ~buf[offset]
        return start_ptr, mask

    def vectorized_store(
        self, store_dtype: DataType, buffer: Expr, source_buffer: Expr, sub_vec_size: int, sub_vec_i: Expr
    ):
        buffer = cast(buffer, ~vectorize(store_dtype, sub_vec_size))
        source_buffer = cast(source_buffer, ~vectorize(store_dtype, sub_vec_size))
        self.buffer_store(buffer, indices=[sub_vec_i], value=source_buffer[sub_vec_i])

    def store(self, inst: StoreGlobalInst, indices: List[Expr], value: Expr, is_first_occurrence: Expr):
        remap = {axis: global_index for axis, global_index in zip(inst.axes, indices)}
        offset: Expr = rewrite(node=inst.offset, rewrite_map=remap)
        mask: Expr = rewrite(node=inst.mask, rewrite_map=remap) if inst.mask is not None else boolean.true
        mask = logical_and(mask, is_first_occurrence)
        buf = cast(inst.ptr, ~inst.inputs[0].dtype) if type_equal(inst.ptr.type, void_p) else inst.ptr
        with self.if_then(mask):
            self.buffer_store(buf=buf, indices=[offset], value=value)


@register_inst_emitter(StoreSharedInst, target=amdgpu_any)
class StoreSharedInstAMDGPUEmitter(StoreInstBaseEmitter):
    def get_buffer_and_mask(self, inst: StoreSharedInst, indices) -> Tuple[Expr, Expr]:
        dst = inst.inputs[0].as_shared_value()
        indices = index_add(inst.offsets, indices)
        remap: Dict[Var, Expr] = {axis: index for axis, index in zip(dst.layout.axes, indices)}
        offset: Expr = rewrite(node=dst.layout.offset, rewrite_map=remap)
        mask: Expr = boolean.true
        buf: Var = self.value2var[dst]
        return ~buf[offset], mask

    def vectorized_store(
        self, store_dtype: DataType, buffer: Expr, source_buffer: Expr, sub_vec_size: int, sub_vec_i: Expr
    ):
        buffer = cast(buffer, ~vectorize(store_dtype, sub_vec_size))
        source_buffer = cast(source_buffer, ~vectorize(store_dtype, sub_vec_size))
        self.buffer_store(buffer, indices=[sub_vec_i], value=source_buffer[sub_vec_i])

    def store(self, inst, indices: List[Expr], value: Expr, is_first_occurrence: Expr):
        buf: Var = self.value2var[inst.inputs[0]]
        indices = index_add(indices, inst.offsets)
        layout: SharedLayout = inst.inputs[0].as_shared_value().layout
        offset: Expr = layout(*indices)
        with self.if_then(is_first_occurrence):
            self.buffer_store(buf, [offset], value=value)
