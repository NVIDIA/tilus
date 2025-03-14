from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Type, Set, no_type_check

from hidet.ir.dtypes import int32, uint8
from hidet.ir.expr import Var, SymbolVar, Expr, cast, logical_not, Constant, convert, tensor_var
from hidet.ir.stmt import DeclareScope
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from hidet.ir.primitives.cuda.vars import threadIdx
from hidet.ir.type import BaseType, void_p
from tilus.utils import prod, cdiv
from tilus.ir.functor import VirtualMachineFunctor
from tilus.ir.inst import Instruction, PrintValueInst, FormatPrintInst
from tilus.ir.program import VirtualMachineProgram
from tilus.ir.stmt import SeqStmt, ForStmt, ForThreadGroupStmt, IfStmt, BreakStmt, WhileStmt
from tilus.ir.tools.instruction_collector import collect_instructions
from tilus.ir.value import Value, SharedValue, RegisterValue, SharedLayout
from tilus.ir.printer import VirtualMachinePrinter
from tilus.target import get_current_target, match_target, gpgpu_any, Target
from tilus.extensions.hidet.ir.expr import convert_sequence
from tilus.extensions.hidet.ir.module import merge_ir_modules
from tilus.extensions.hidet.ir.builders import StmtBuilder, FunctionBuilder
from tilus.extensions.hidet.ir.tools import rewrite
from tilus.ir.weight_transform import (
    WeightTransform,
    WeightLayoutTransformGeneric,
    WeightValueTransform,
    WeightLayoutTransform,
)


class InvalidInstruction(Exception):
    def __init__(self, inst):
        self.inst = inst


class CodeGenerationFailed(Exception):
    pass


def is_nvgpu():
    return get_current_target().is_nvgpu()


def is_amdgpu():
    return get_current_target().is_amdgpu()


class BaseInstEmitter(StmtBuilder):
    # inst -> emitter
    REGISTRY: Dict[Type[Instruction], Dict[Target, Type["BaseInstEmitter"]]] = {}

    def __init__(self, codegen) -> None:
        super().__init__()
        self.codegen: MainKernelCodegen = codegen

    def sync(self):
        from hidet.ir.primitives.cuda import syncthreads

        if self.codegen.thread_groups.num_levels() == 1:  # all threads in the cta
            self.append(syncthreads())
        else:
            if get_current_target().is_nvgpu():
                from tilus.extensions.hidet.ir.primitives.cuda.barrier import barrier_sync

                barrier = self.codegen.thread_groups.num_levels() - 1
                count = self.codegen.thread_groups.group_size[-1]
                self.append(barrier_sync(barrier=barrier, count=count))
            else:
                raise NotImplementedError()

    def sync_reduce(self, value: Expr, op: str) -> Expr:
        if get_current_target().is_nvgpu():
            from hidet.ir.primitives.cuda.sync import syncthreads_and, syncthreads_or
            from tilus.extensions.hidet.ir.primitives.cuda.barrier import barrier_sync

            op2sync = {"and": syncthreads_and, "or": syncthreads_or}
            syncthreads_op = op2sync[op]

            if self.codegen.thread_groups.num_levels() == 1:  # all threads in the cta
                return syncthreads_op(value)
            else:
                barrier = self.codegen.thread_groups.num_levels() - 1
                count = self.codegen.thread_groups.group_size[-1]
                self.append(barrier_sync(barrier=barrier, count=count))
                raise NotImplementedError("barrier_sync_reduce")
        else:
            raise NotImplementedError()

    def get_or_allocate_var(self, value: Value, name: Optional[str] = None) -> Var:
        if value in self.value2var:
            return self.value2var[value]
        else:
            if isinstance(value, RegisterValue):
                name = name if name else "regs"
                var = self.declare(tensor_var(name, shape=[value.size], dtype=value.dtype), scope=DeclareScope.Register)
            elif isinstance(value, SharedValue):
                name = name if name else "smem"
                var = self.declare(tensor_var(name, shape=[value.size], dtype=value.dtype), scope=DeclareScope.Shared)
            else:
                raise NotImplementedError()
            self.value2var[value] = var
            return var

    @property
    def current_worker(self) -> Expr:
        return self.codegen.thread_groups.current_worker[-1]

    @property
    def thread_groups(self):
        return self.codegen.thread_groups

    @property
    def value2var(self) -> Dict[Value, Var]:
        return self.codegen.value2var

    @property
    def shared_value_shared_space_addr(self):
        return self.codegen.shared_value_shared_space_addr

    @property
    def num_warps(self) -> int:
        assert self.codegen._program is not None
        return self.codegen._program.num_warps

    def emit(self, inst: Instruction):
        raise NotImplementedError()


def register_inst_emitter(inst_cls: Type[Instruction], *, target: Optional[Target] = None):
    assert issubclass(inst_cls, Instruction)
    if target is None:
        target = gpgpu_any

    def decorator(emitter_cls: Type[BaseInstEmitter]):
        assert issubclass(emitter_cls, BaseInstEmitter)

        if inst_cls not in BaseInstEmitter.REGISTRY:
            BaseInstEmitter.REGISTRY[inst_cls] = {}

        if target in BaseInstEmitter.REGISTRY[inst_cls]:
            raise ValueError(f"Emitter for instruction {inst_cls} and target {target} already exists")

        BaseInstEmitter.REGISTRY[inst_cls][target] = emitter_cls
        return emitter_cls

    return decorator


def resolve_inst_emitter(inst_cls: Type[Instruction]) -> Optional[Type[BaseInstEmitter]]:
    target = get_current_target()
    emitter_classes = BaseInstEmitter.REGISTRY.get(inst_cls, {})
    matched_target = match_target(target, list(emitter_classes))
    if matched_target is None:
        return None
    return emitter_classes[matched_target]


class SharedMemoryAllocator:
    def __init__(self) -> None:
        self.free_slots: List[Tuple[int, int]] = [(0, (1 << 32) - 1)]
        self.addr2nbytes: Dict[int, int] = {}
        self.allocated: int = 0
        self.maximum_allocated: int = 0

    def allocate(self, nbytes: int) -> int:
        # align the nbytes to 128 bytes aligned
        nbytes = (nbytes + 127) // 128 * 128

        # find the first slot that can fit the request
        i = min(i for i, (start, end) in enumerate(self.free_slots) if end - start >= nbytes)
        addr = self.free_slots[i][0]
        if self.free_slots[i][1] - self.free_slots[i][0] == nbytes:
            # remove the slot
            del self.free_slots[i]
        else:
            # shrink the slot
            self.free_slots[i] = (addr + nbytes, self.free_slots[i][1])
        self.addr2nbytes[addr] = nbytes
        self.maximum_allocated = max(self.maximum_allocated, addr + nbytes)
        self.allocated += nbytes
        return addr

    def free(self, addr: int):
        # find the slot that is right before the address
        before = [i for i, slot in enumerate(self.free_slots) if slot[1] <= addr]
        after = [i for i, slot in enumerate(self.free_slots) if slot[0] > addr]
        assert len(before) + len(after) == len(self.free_slots)
        nbytes = self.addr2nbytes[addr]
        if (
            before
            and after
            and self.free_slots[before[-1]][1] == addr
            and self.free_slots[after[0]][0] == addr + nbytes
        ):
            # merge three slots
            self.free_slots[before[-1]] = (self.free_slots[before[-1]][0], self.free_slots[after[0]][1])
        elif before and self.free_slots[before[-1]][1] == addr:
            # merge with previous slot
            self.free_slots[before[-1]] = (self.free_slots[before[-1]][0], addr + nbytes)
        elif after and self.free_slots[after[0]][0] == addr + nbytes:
            # merge with next slot
            self.free_slots[after[0]] = (addr, self.free_slots[after[0]][1])
        else:
            # add a new slot
            self.free_slots.append((addr, addr + nbytes))
            self.free_slots = list(sorted(self.free_slots, key=lambda x: x[0]))
        self.allocated -= nbytes
        del self.addr2nbytes[addr]


class VirtualMachineCodegen(VirtualMachineFunctor):
    def __init__(self):
        super().__init__()
        self.weight_transform_codegen = WeightTransformKernelCodegen()
        self.main_kernel_codegen = MainKernelCodegen()

    def visit_Program(self, prog: VirtualMachineProgram):
        ir_modules = [self.weight_transform_codegen(prog), self.main_kernel_codegen(prog)]
        return merge_ir_modules(ir_modules)


class MainKernelCodegen(VirtualMachineFunctor):
    GMEM_WORKSPACE_NAME = "__gmem_workspace"
    GMEM_CLEAN_WORKSPACE_NAME = "__gmem_clean_workspace"

    @dataclass
    class ThreadGroups:
        current_worker: List[Expr]
        num_groups: List[int]
        group_size: List[int]

        def num_levels(self):
            return len(self.num_groups)

    def __init__(self) -> None:
        super().__init__()
        self._builder: Optional[FunctionBuilder] = None
        self._program: Optional[VirtualMachineProgram] = None
        self.printer: VirtualMachinePrinter = VirtualMachinePrinter()

        # value mapping
        self.value2var: Dict[Value, Var] = {}

        # global memory management
        self.gmem_base_ptr: Var = SymbolVar(dtype=~uint8, name=self.GMEM_WORKSPACE_NAME)  # type: ignore # todo: update hidet to allow SymbolVar supports pointer
        self.gmem_allocated: Expr = int32.zero
        self.gmem_maximum_allocated: Expr = int32.zero
        self.gmem_clean_base_ptr: Var = SymbolVar(dtype=~uint8, name=self.GMEM_CLEAN_WORKSPACE_NAME)  # type: ignore
        self.gmem_clean_allocated: Expr = int32.zero
        self.gmem_clean_maximum_allocated: Expr = int32.zero

        # shared memory allocator
        self.smem_allocator: SharedMemoryAllocator = SharedMemoryAllocator()
        # mapping from shared value to the address in shared memory allocator for all allocated shared values
        self.shared_value_allocator_addr: Dict[SharedValue, int] = {}
        # mapping from shared value to the address in shared memory space (e.g., returned by cvta ptx instruction)
        self.shared_value_shared_space_addr: Dict[SharedValue, Var] = {}

        # shared memory workspace
        self.smem_workspace: Optional[SharedValue] = None

        # stacks of for_thread_groups
        self.thread_groups = MainKernelCodegen.ThreadGroups([], [], [])

    def __call__(self, prog: VirtualMachineProgram):
        return self.visit(prog)

    @property
    def program(self) -> VirtualMachineProgram:
        assert self._program is not None
        return self._program

    @property
    def builder(self) -> FunctionBuilder:
        assert self._builder is not None
        return self._builder

    def sync(self):
        from tilus.ir.inst import SyncThreadsInst

        self.visit(SyncThreadsInst())

    def allocate_shared_value(self, value: SharedValue, nbytes: int):
        addr: int = self.smem_allocator.allocate(nbytes)
        assert value not in self.shared_value_allocator_addr
        self.shared_value_allocator_addr[value] = addr
        return addr

    def free_shared_value(self, value: SharedValue):
        assert value in self.shared_value_allocator_addr
        self.smem_allocator.free(addr=self.shared_value_allocator_addr[value])
        del self.shared_value_allocator_addr[value]

    def allocate_global_memory(self, nbytes: Expr, clean: bool) -> Expr:
        nbytes = (nbytes + 127) // 128 * 128  # align to 128 bytes
        if clean:
            ret = self.gmem_clean_base_ptr + self.gmem_clean_allocated
            self.gmem_clean_allocated = self.gmem_clean_allocated + nbytes
            self.gmem_clean_maximum_allocated = self.gmem_clean_allocated
        else:
            ret = self.gmem_base_ptr + self.gmem_allocated
            self.gmem_allocated = self.gmem_allocated + nbytes
            self.gmem_maximum_allocated = self.gmem_allocated
        return ret

    def check_emitter_existence(self) -> None:
        failed_instructions: Set[str] = set()
        for inst in collect_instructions(self.program):
            if resolve_inst_emitter(inst.__class__) is None:
                failed_instructions.add(inst.__class__.__name__)

        if failed_instructions:
            raise CodeGenerationFailed(
                "Failed to find emitter for the following instructions: \n{}".format("\n".join(failed_instructions))
            )

    def init_block_axes(self):
        with self.builder.if_then(logical_not(self._program.block_mapping.predicate)):
            self.builder.ret()
        for axis, value in self._program.block_mapping.virtual_axes_values.items():
            self.builder.declare(v=axis, init=value)

    def init_smem_workspace(self, program: VirtualMachineProgram):
        smem_workspace_nbytes: int = 0
        for inst in collect_instructions(program):
            smem_workspace_nbytes = max(smem_workspace_nbytes, inst.request_shared_workspace())
        if smem_workspace_nbytes > 0:
            value = SharedValue(dtype=uint8, layout=SharedLayout.repeat(smem_workspace_nbytes))
            self.allocate_shared_value(value, nbytes=smem_workspace_nbytes)
            self.value2var[value] = self.builder.declare(
                v=Var("temp_smem", type=void_p),
                init=dynamic_shared_memory(byte_offset=self.shared_value_allocator_addr[value], dtype=uint8),
            )
            self.smem_workspace = value

    def generate_launch_function(self, ir_module: IRModule, kernel_func: Function) -> IRModule:
        from tilus.extensions.hidet.ir.primitives.runtime import set_symbol_value_ptr
        from hidet.transforms.generate_launch_func import add_launch_func
        from hidet.ir.stmt import SeqStmt

        add_launch_func(ir_module, kernel_func=kernel_func)

        launch_func = ir_module.functions["launch"]

        if is_nvgpu():
            from hidet.ir.primitives.runtime import request_cuda_workspace

            request_workspace = request_cuda_workspace
        elif is_amdgpu():
            from hidet.ir.primitives.runtime import request_hip_workspace

            request_workspace = request_hip_workspace
        else:
            assert False

        # set the workspace
        sb = StmtBuilder()
        remap = {prog_param: launch_param for prog_param, launch_param in zip(self.program.params, launch_func.params)}
        if not (isinstance(self.gmem_allocated, Constant) and int(self.gmem_allocated) == 0):
            sb += set_symbol_value_ptr(
                self.GMEM_WORKSPACE_NAME,
                cast(
                    request_workspace(nbytes=rewrite(self.gmem_maximum_allocated, remap), require_clean=False), ~uint8
                ),
            )
        if not (isinstance(self.gmem_clean_allocated, Constant) and int(self.gmem_clean_allocated) == 0):
            sb += set_symbol_value_ptr(
                self.GMEM_CLEAN_WORKSPACE_NAME,
                cast(
                    request_workspace(nbytes=rewrite(self.gmem_clean_maximum_allocated, remap), require_clean=True),
                    ~uint8,
                ),
            )

        launch_func.body = SeqStmt([sb.finish(), launch_func.body])
        return ir_module

    def visit_Program(self, prog: VirtualMachineProgram):
        # warmup printer
        self.printer(prog)

        self._program = prog

        self.check_emitter_existence()

        self._builder = FunctionBuilder(
            name=prog.name,
            kind="cuda_kernel" if is_nvgpu() else "hip_kernel",
            label="",
            grid_dim=self._program.block_mapping.hardware_num_blocks,
            block_dim=prog.num_warps * 32,
            dynamic_smem_bytes=None,
            min_blocks=None,
        )
        self.builder.extend_params(prog.params)

        # init for_thread_group stack
        self.thread_groups.num_groups = [1]
        self.thread_groups.group_size = [prog.num_warps * 32]
        self.thread_groups.current_worker = [threadIdx.x]

        # init pre-defined variables
        self.init_block_axes()
        self.init_smem_workspace(prog)

        # emit body
        self.visit(prog.body)

        # check shared memory allocation and set dynamic shared memory size
        if self.smem_workspace:
            self.free_shared_value(self.smem_workspace)
            self.smem_workspace = None
        if self.smem_allocator.allocated != 0:
            raise ValueError("Shared memory is not properly allocated/freed")
        if self.smem_allocator.maximum_allocated > get_current_target().properties.shared_memory_per_block:
            raise CodeGenerationFailed(
                "Request shared memory {} bytes, but the device only allows {} bytes.".format(
                    self.smem_allocator.maximum_allocated, get_current_target().properties.shared_memory_per_block
                )
            )
        if is_nvgpu():
            self.builder.attrs["cuda.dynamic_smem_bytes"] = self.smem_allocator.maximum_allocated
        elif is_amdgpu():
            self.builder.attrs["hip.dynamic_smem_bytes"] = self.smem_allocator.maximum_allocated
        else:
            assert False

        self.builder.finish_func()
        kernel_function = self.builder.get()
        ir_module = IRModule(functions={kernel_function.name: kernel_function})
        ir_module = self.generate_launch_function(ir_module, kernel_func=kernel_function)
        return ir_module

    def visit_SeqStmt(self, stmt: SeqStmt):
        for sub_stmt in stmt.seq:
            self.visit(sub_stmt)

    def visit_IfStmt(self, stmt: IfStmt):
        with self.builder.if_then(stmt.cond):
            self.visit(stmt.then_body)
        if stmt.else_body is not None:
            with self.builder.otherwise():
                self.visit(stmt.else_body)

    def visit_WhileStmt(self, stmt: WhileStmt):
        with self.builder.while_loop(stmt.cond):
            self.visit(stmt.body)

    def visit_ForStmt(self, stmt: ForStmt):
        if stmt.unroll_factor is None:
            attr = "."
        elif stmt.unroll_factor == -1:
            attr = "u"
        else:
            attr = "u{}".format(stmt.unroll_factor)  # no unroll
        with self.builder.for_loop(stmt.iter_var, stmt.extent, attr=attr):
            self.visit(stmt.body)

    def visit_ForThreadGroupStmt(self, stmt: ForThreadGroupStmt):
        prev_group_size = self.thread_groups.group_size[-1]
        group_size = prev_group_size // stmt.num_groups

        self.builder.declare(v=stmt.iter_var, init=threadIdx.x % prev_group_size // group_size)
        with self.builder.for_range(stmt.num_groups) as i:
            self.thread_groups.num_groups.append(stmt.num_groups)
            self.thread_groups.group_size.append(group_size)
            self.thread_groups.current_worker.append(threadIdx.x % group_size)
            with self.builder.if_then(stmt.iter_var == i):
                self.visit(stmt.body)
            self.thread_groups.group_size.pop()
            self.thread_groups.num_groups.pop()
            self.thread_groups.current_worker.pop()

            self.sync()

    def visit_BreakStmt(self, stmt: BreakStmt):
        self.builder.brk()

    def visit_Instruction(self, inst: Instruction):
        # insert a comment statement
        skip_comment_instructions = (PrintValueInst, FormatPrintInst)
        if not isinstance(inst, skip_comment_instructions):
            self.builder.comment(str(self.printer(inst)), style="/*")

        # implement the vm instruction
        emitter_cls = resolve_inst_emitter(inst.__class__)
        if emitter_cls is None:
            raise RuntimeError("Can not resolve the emitter for instruction: {}".format(inst.__class__.__name__))
        emitter = emitter_cls(self)
        emitter.emit(inst)
        self.builder.append(emitter.finish())


class WeightTransformKernelCodegen(VirtualMachineFunctor):
    def __init__(self) -> None:
        super().__init__()
        self.ir_module = IRModule()
        self.param_to_apply_kernels: Dict[Var, List[Var]] = {}
        self.param_to_reverse_kernels: Dict[Var, List[Var]] = {}

    def visit_Program(self, prog: VirtualMachineProgram) -> IRModule:
        # generate weight transform functions and reverse transform functions for each weight param
        for param, transforms in prog.weight_transforms.items():
            self.param_to_apply_kernels[param] = []
            self.param_to_reverse_kernels[param] = []
            for idx, transform in enumerate(transforms):
                apply_kernel, reverse_kernel = self.generate_transform(param, idx, transform)
                self.param_to_apply_kernels[param].append(apply_kernel)
                self.param_to_reverse_kernels[param].append(reverse_kernel)

        # generate driver functions
        self.generate_driver_functions(prog)

        return self.ir_module

    def generate_driver_functions(self, prog) -> None:
        import hidet.ir.primitives.runtime
        from hidet.lang import attrs, meta

        kernel_param_types: List[BaseType] = [param.type for param in prog.params]
        num_weights: int = len(prog.weight_transforms)
        weight2nbytes: Dict[Var, int] = {}
        for param in prog.weight_transforms.keys():
            nbytes: Optional[int] = prog.param2attrs[param].weight_nbytes
            if nbytes is None:
                raise ValueError("Weight transform requires weight_nbytes to be set, got {}".format(weight2nbytes))
            weight2nbytes[param] = nbytes

        weight2arg_idx: Dict[Var, int] = {param: i for i, param in enumerate(prog.params) if param in weight2nbytes}
        weight_params: List[Var] = list(weight2nbytes.keys())
        workspace_size: int = max(weight2nbytes.values()) if num_weights > 0 else 0

        request_workspace = (
            hidet.ir.primitives.runtime.request_cuda_workspace
            if get_current_target().is_nvgpu()
            else hidet.ir.primitives.runtime.request_hip_workspace
        )
        memcpy_async = hidet.ir.primitives.cuda.memcpy_async if is_nvgpu() else hidet.ir.primitives.hip.memcpy_async
        kind = "cuda_to_cuda" if is_nvgpu() else "hip_to_hip"

        @no_type_check
        @hidet.script
        def apply_weight_transforms(args: meta.types(kernel_param_types)):
            attrs.func_kind = "public"

            if num_weights > 0:
                workspace = request_workspace(nbytes=workspace_size)
                for weight_param in meta.each(weight_params):
                    weight_arg = args[weight2arg_idx[weight_param]]
                    weight_nbytes = weight2nbytes[weight_param]
                    for transform in meta.each(self.param_to_apply_kernels[weight_param]):
                        transform(workspace, weight_arg)
                        memcpy_async(dst=weight_arg, src=workspace, count=convert(weight_nbytes), kind=kind)

        @no_type_check
        @hidet.script  # type: ignore
        def reverse_weight_transforms(args: meta.types(kernel_param_types)):  # type: ignore
            attrs.func_kind = "public"

            if num_weights > 0:
                workspace = request_workspace(nbytes=workspace_size)
                for weight_param in meta.each(weight_params):
                    weight_arg = args[weight2arg_idx[weight_param]]
                    weight_nbytes = weight2nbytes[weight_param]
                    for transform in meta.each(reversed(self.param_to_reverse_kernels[weight_param])):
                        transform(workspace, weight_arg)
                        memcpy_async(dst=weight_arg, src=workspace, count=convert(weight_nbytes), kind=kind)

        assert isinstance(apply_weight_transforms, Function) and isinstance(reverse_weight_transforms, Function)
        self.ir_module.add_function(apply_weight_transforms.name, apply_weight_transforms)
        self.ir_module.add_function(reverse_weight_transforms.name, reverse_weight_transforms)

    def generate_transform(self, param, idx: int, wt: WeightTransform) -> Tuple[Var, Var]:
        if isinstance(wt, WeightLayoutTransform):
            return self.generate_layout_transform(param, idx, wt)
        elif isinstance(wt, WeightLayoutTransformGeneric):
            return self.generate_layout_transform_generic(param, idx, wt)
        elif isinstance(wt, WeightValueTransform):
            return self.generate_value_transform(param, idx, wt)
        else:
            raise NotImplementedError(wt.__class__.__name__)

    def generate_layout_transform(self, param: Var, idx: int, transform: WeightLayoutTransform) -> Tuple[Var, Var]:
        from tilus.ir.builder import VirtualMachineBuilder
        from tilus.extensions.hidet.ir.utils.index_transform import (
            index_add,
            index_multiply,
            index_sum,
            index_serialize,
        )

        func_name = "layout_transform_{}_{}".format(param.hint, idx)

        dtype = transform.dtype
        original_layout = transform.original_layout
        transformed_layout = transform.transformed_layout
        transformed_dtype = transform.transformed_dtype
        tile_shape = transform.tile_shape
        num_tiles: List[Expr] = list(convert_sequence(transform.num_tiles))
        strides: List[int] = transform.strides
        assert original_layout.num_workers % 32 == 0
        num_warps = original_layout.num_workers // 32

        vb = VirtualMachineBuilder()

        # generate transform program
        with vb.program(name="apply_" + func_name, num_warps=num_warps, params={"dst": void_p, "src": void_p}) as (
            dst,
            src,
        ):
            block_axes: List[Var] = vb.virtual_blocks(num_blocks=num_tiles)
            loaded = vb.load_global(
                dtype=dtype,
                layout=original_layout,
                ptr=src,
                f_offset=lambda axes: index_sum(
                    index_multiply(index_add(index_multiply(block_axes, tile_shape), axes), strides)
                ),
            )
            viewed = vb.view(loaded, dtype=transformed_dtype, layout=transformed_layout)
            vb.store_global(
                viewed,
                ptr=dst,
                f_offset=lambda axes: index_serialize(block_axes, shape=num_tiles) * transformed_layout.shape[0]
                + axes[0],
            )

        apply_program = vb.flush_program()

        # generate reverse transform program
        with vb.program(name="reverse_" + func_name, num_warps=num_warps, params={"dst": void_p, "src": void_p}) as (
            dst,
            src,
        ):
            block_axes = vb.virtual_blocks(num_blocks=num_tiles)
            loaded = vb.load_global(
                dtype=transformed_dtype,
                layout=transformed_layout,
                ptr=src,
                f_offset=lambda axes: index_serialize(block_axes, shape=num_tiles) * transformed_layout.shape[0]
                + axes[0],
            )
            viewed = vb.view(loaded, dtype=dtype, layout=original_layout)
            vb.store_global(
                viewed,
                ptr=dst,
                f_offset=lambda axes: index_sum(
                    index_multiply(index_add(index_multiply(block_axes, tile_shape), axes), strides)
                ),
            )

        reverse_program = vb.flush_program()

        # build the two programs and extract the functions
        apply_ir_module: IRModule = generate_ir_module(apply_program)
        reverse_ir_module: IRModule = generate_ir_module(reverse_program)

        results: List[Var] = []
        for name, module in [["apply_" + func_name, apply_ir_module], ["reverse_" + func_name, reverse_ir_module]]:
            self.ir_module.add_function(name, module.functions[name])
            results.append(self.ir_module.lookup_var(name))

        return results[0], results[1]

    def generate_layout_transform_generic(
        self, param: Var, idx: int, transform: WeightLayoutTransformGeneric
    ) -> Tuple[Var, Var]:
        from hidet.lang import attrs, script
        from hidet.lang.types import void_p, tensor_pointer

        if is_nvgpu():
            from hidet.lang.cuda import blockIdx, threadIdx, blockDim
        else:
            # todo: make cuda/hip share the same dim3
            from hidet.lang.hip import blockIdx, threadIdx, blockDim  # type: ignore[assignment]

        dtype = transform.dtype
        func_name = "layout_transform_{}_{}".format(param.hint, idx)
        num_elements = transform.size

        if dtype.nbits < 8:
            # hidet support to sub-byte type is not complete
            # we avoid generating kernels for sub-byte types for now
            assert num_elements * dtype.nbits % 8 == 0

            nbits = num_elements * dtype.nbits
            nbytes = nbits // 8
            num_lanes: int = 8 // dtype.nbits
            lane_bits = dtype.nbits
            lane_mask = (1 << lane_bits) - 1
            block_dim = 128
            grid_dim = cdiv(nbytes, block_dim)

            @no_type_check
            @script
            def apply_kernel(dst: void_p, src: void_p):
                attrs.func_kind = "gpgpu_kernel"
                attrs.func_name = "apply_" + func_name
                attrs.gpgpu.grid_dim = grid_dim
                attrs.gpgpu.block_dim = block_dim

                byte_index = blockIdx.x * blockDim.x + threadIdx.x

                dst_uint8 = tensor_pointer(uint8, shape=[nbytes], init=cast(dst, ~uint8))
                src_uint8 = tensor_pointer(uint8, shape=[nbytes], init=cast(src, ~uint8))

                if byte_index < nbytes:
                    value: uint8 = uint8.zero

                    for lane in range(num_lanes):
                        out_element_index = byte_index * num_lanes + lane
                        in_element_index = transform.mapping(out_element_index)
                        in_byte_index = in_element_index // num_lanes
                        in_lane = in_element_index % num_lanes
                        in_byte = src_uint8[in_byte_index]
                        value = value | ((in_byte >> (in_lane * lane_bits) & lane_mask) << (lane * lane_bits))
                    dst_uint8[byte_index] = value

            @no_type_check
            @script
            def reverse_kernel(dst: void_p, src: void_p):
                attrs.func_kind = "gpgpu_kernel"
                attrs.func_name = "reverse_" + func_name
                attrs.gpgpu.grid_dim = grid_dim
                attrs.gpgpu.block_dim = block_dim

                i = blockIdx.x * blockDim.x + threadIdx.x

                dst_uint8 = tensor_pointer(uint8, shape=[nbytes], init=cast(dst, ~uint8))
                src_uint8 = tensor_pointer(uint8, shape=[nbytes], init=cast(src, ~uint8))

                if i < nbytes:
                    value: uint8 = uint8.zero

                    for lane in range(num_lanes):
                        out_element_index = i * num_lanes + lane
                        in_element_index = transform.reverse_mapping(out_element_index)
                        in_byte_index = in_element_index // num_lanes
                        in_lane = in_element_index % num_lanes
                        in_byte = src_uint8[in_byte_index]
                        value = value | ((in_byte >> (in_lane * lane_bits) & lane_mask) << (lane * lane_bits))
                    dst_uint8[i] = value

        else:
            block_dim = 128
            grid_dim = cdiv(num_elements, block_dim)

            @no_type_check
            @script
            def apply_kernel(dst: void_p, src: void_p):
                attrs.func_kind = "gpgpu_kernel"
                attrs.func_name = "apply_" + func_name
                attrs.gpgpu.grid_dim = grid_dim
                attrs.gpgpu.block_dim = block_dim

                i = blockIdx.x * blockDim.x + threadIdx.x

                typed_dst = tensor_pointer(dtype, shape=[num_elements], init=cast(dst, ~dtype))
                typed_src = tensor_pointer(dtype, shape=[num_elements], init=cast(src, ~dtype))

                if i < num_elements:
                    out_index = i
                    in_index = transform.mapping(out_index)
                    typed_dst[out_index] = typed_src[in_index]

            @no_type_check
            @script
            def reverse_kernel(dst: void_p, src: void_p):
                attrs.func_kind = "gpgpu_kernel"
                attrs.func_name = "reverse_" + func_name
                attrs.gpgpu.grid_dim = grid_dim
                attrs.gpgpu.block_dim = block_dim

                i = blockIdx.x * blockDim.x + threadIdx.x

                typed_dst = tensor_pointer(dtype, shape=[num_elements], init=cast(dst, ~dtype))
                typed_src = tensor_pointer(dtype, shape=[num_elements], init=cast(src, ~dtype))

                if i < num_elements:
                    out_index = i
                    in_index = transform.reverse_mapping(out_index)
                    typed_dst[out_index] = typed_src[in_index]

        assert isinstance(apply_kernel, Function) and isinstance(reverse_kernel, Function)
        self.ir_module.add_function(apply_kernel.name, apply_kernel)
        self.ir_module.add_function(reverse_kernel.name, reverse_kernel)

        return self.ir_module.lookup_var(apply_kernel.name), self.ir_module.lookup_var(reverse_kernel.name)

    def generate_value_transform(self, param: Var, idx: int, transform: WeightValueTransform) -> Tuple[Var, Var]:
        from hidet.lang import attrs, script
        from hidet.lang.types import void_p, tensor_pointer
        from hidet.lang.cuda import blockIdx, threadIdx, blockDim

        dtype = transform.dtype
        func_name = "value_transform_{}_{}".format(param.hint, idx)
        num_elements = prod(transform.shape)
        block_dim = 128
        grid_dim = cdiv(num_elements, block_dim)

        @no_type_check
        @script
        def apply_kernel(dst: void_p, src: void_p):
            attrs.func_kind = "cuda_kernel"
            attrs.func_name = "apply_" + func_name
            attrs.cuda.grid_dim = grid_dim
            attrs.cuda.block_dim = block_dim

            typed_dst = tensor_pointer(dtype, shape=[num_elements], init=cast(dst, ~dtype))
            typed_src = tensor_pointer(dtype, shape=[num_elements], init=cast(src, ~dtype))

            i = blockIdx.x * blockDim.x + threadIdx.x

            if i < num_elements:
                typed_dst[i] = transform.mapping(typed_src[i])

        @no_type_check
        @script
        def reverse_kernel(dst: void_p, src: void_p):
            attrs.func_kind = "cuda_kernel"
            attrs.func_name = "reverse_" + func_name
            attrs.cuda.grid_dim = grid_dim
            attrs.cuda.block_dim = block_dim

            typed_dst = tensor_pointer(dtype, shape=[num_elements], init=cast(dst, ~dtype))
            typed_src = tensor_pointer(dtype, shape=[num_elements], init=cast(src, ~dtype))

            i = blockIdx.x * blockDim.x + threadIdx.x

            if i < num_elements:
                typed_dst[i] = transform.reverse_mapping(typed_src[i])

        assert isinstance(apply_kernel, Function) and isinstance(reverse_kernel, Function)
        self.ir_module.add_function(apply_kernel.name, apply_kernel)
        self.ir_module.add_function(reverse_kernel.name, reverse_kernel)

        return self.ir_module.lookup_var(apply_kernel.name), self.ir_module.lookup_var(reverse_kernel.name)


def generate_ir_module(prog: VirtualMachineProgram) -> IRModule:
    codegen = VirtualMachineCodegen()
    ir_module: IRModule = codegen(prog)
    ir_module.attrs["default_kernel"] = prog.name
    return ir_module
