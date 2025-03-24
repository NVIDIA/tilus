from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple, Type

from hidet.ir.dtypes import int32, uint8
from hidet.ir.expr import Constant, Expr, SymbolVar, Var, cast, tensor_pointer_var, tensor_var
from hidet.ir.func import Function as HidetFunction
from hidet.ir.module import IRModule
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from hidet.ir.primitives.cuda.vars import threadIdx
from hidet.ir.stmt import DeclareScope
from hidet.ir.type import void_p
from tilus.extensions.hidet.ir.builders import FunctionBuilder, StmtBuilder
from tilus.extensions.hidet.ir.module import merge_ir_modules
from tilus.extensions.hidet.ir.tools import rewrite
from tilus.ir.func import Function
from tilus.ir.functors import IRFunctor
from tilus.ir.inst import Instruction
from tilus.ir.instructions import FormatPrintInst, PrintTensorInst
from tilus.ir.prog import Program
from tilus.ir.stmt import (
    AssignStmt,
    BreakStmt,
    DeclareStmt,
    ForStmt,
    ForThreadGroupStmt,
    IfStmt,
    InstStmt,
    SeqStmt,
    TensorPtrStmt,
    WhileStmt,
)
from tilus.ir.tensor import GlobalTensor, RegisterTensor, SharedLayout, SharedTensor, Tensor
from tilus.ir.tools import IRPrinter
from tilus.ir.tools.instruction_collector import collect_instructions
from tilus.target import Target, get_current_target, gpgpu_any, match_target


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

    def __init__(self, codegen: Codegen) -> None:
        super().__init__()
        # todo: currently, the instruction emitters (that inherit from BaseInstEmitter) directly access the codegen
        #       object to access some data in the codegen object. This is not a good design. We should refactor this
        #       to use the methods of the BaseInstEmitter class to access the data in the codegen object.
        self.codegen: Codegen = codegen

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

    def get_or_allocate_var(self, tensor: Tensor, name: Optional[str] = None) -> Var:
        if tensor in self.tensor2var:
            return self.tensor2var[tensor]
        else:
            if isinstance(tensor, RegisterTensor):
                name = name if name else "regs"
                var = self.declare(
                    tensor_var(name, shape=[tensor.local_size], dtype=tensor.dtype), scope=DeclareScope.Register
                )
            elif isinstance(tensor, SharedTensor):
                name = name if name else "smem"
                var = self.declare(tensor_pointer_var(name, shape=[tensor.size], dtype=tensor.dtype))
            elif isinstance(tensor, GlobalTensor):
                name = name if name else "gmem"
                var = self.declare(tensor_pointer_var(name, shape=[tensor.size], dtype=tensor.dtype))
            else:
                raise NotImplementedError()
            self.tensor2var[tensor] = var
            return var

    @property
    def current_worker(self) -> Expr:
        return self.codegen.thread_groups.current_worker[-1]

    @property
    def thread_groups(self):
        return self.codegen.thread_groups

    @property
    def tensor2var(self) -> Dict[Tensor, Var]:
        return self.codegen.tensor2var

    @property
    def shared_value_shared_space_addr(self):
        return self.codegen.shared_value_shared_space_addr

    @property
    def num_warps(self) -> int:
        return self.codegen.program.num_warps

    def emit(self, inst: Instruction) -> None:
        raise NotImplementedError()


def register_emitter(
    inst_cls: Type[Instruction], *, target: Optional[Target] = None
) -> Callable[[Type[BaseInstEmitter]], Type[BaseInstEmitter]]:
    assert issubclass(inst_cls, Instruction)
    if target is None:
        target = gpgpu_any

    def decorator(emitter_cls: Type[BaseInstEmitter]) -> Type[BaseInstEmitter]:
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

    def free(self, addr: int) -> None:
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


class Codegen(IRFunctor):
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
        self._program: Optional[Function] = None
        self.printer: IRPrinter = IRPrinter()

        # value mapping
        self.tensor2var: Dict[Tensor, Var] = {}

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
        self.shared_value_allocator_addr: Dict[SharedTensor, int] = {}
        # mapping from shared value to the address in shared memory space (e.g., returned by cvta ptx instruction)
        self.shared_value_shared_space_addr: Dict[SharedTensor, Var] = {}

        # shared memory workspace
        self.smem_workspace: Optional[SharedTensor] = None

        # stacks of for_thread_groups
        self.thread_groups = Codegen.ThreadGroups([], [], [])

    def __call__(self, prog: Function) -> IRModule:
        return self.visit(prog)

    @property
    def program(self) -> Function:
        assert self._program is not None
        return self._program

    @property
    def builder(self) -> FunctionBuilder:
        assert self._builder is not None
        return self._builder

    def sync(self) -> None:
        from tilus.ir.inst import SyncThreadsInst

        self.visit(SyncThreadsInst.create())

    def allocate_shared_value(self, value: SharedTensor, nbytes: int) -> int:
        addr: int = self.smem_allocator.allocate(nbytes)
        assert value not in self.shared_value_allocator_addr
        self.shared_value_allocator_addr[value] = addr
        return addr

    def free_shared_value(self, value: SharedTensor) -> None:
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

    def init_smem_workspace(self, program: Function) -> None:
        smem_workspace_nbytes: int = 0
        # for inst in collect_instructions(program):    # todo: add this to emiter
        #     smem_workspace_nbytes = max(smem_workspace_nbytes, inst.request_shared_workspace())
        if smem_workspace_nbytes > 0:
            value = SharedTensor.create(dtype=uint8, layout=SharedLayout.repeat(smem_workspace_nbytes))
            self.allocate_shared_value(value, nbytes=smem_workspace_nbytes)
            self.tensor2var[value] = self.builder.declare(
                v=Var("temp_smem", type=void_p),
                init=dynamic_shared_memory(byte_offset=self.shared_value_allocator_addr[value], dtype=uint8),
            )
            self.smem_workspace = value

    def generate_launch_function(self, ir_module: IRModule, kernel_func: HidetFunction) -> IRModule:
        from hidet.ir.stmt import SeqStmt
        from hidet.transforms.generate_launch_func import add_launch_func
        from tilus.extensions.hidet.ir.primitives.runtime import set_symbol_value_ptr

        add_launch_func(ir_module, kernel_func=kernel_func)

        launch_func = ir_module.functions["launch"]
        launch_func = HidetFunction(
            name=kernel_func.name.removesuffix("_kernel"),
            params=launch_func.params,
            body=launch_func.body,
            ret_type=launch_func.ret_type,
            kind=launch_func.kind,
            attrs=launch_func.attrs,
        )

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
        updated_ir_module = IRModule(
            functions={
                launch_func.name: launch_func,
                kernel_func.name: kernel_func,
            },
        )
        return updated_ir_module

    def visit_Function(self, func: Function) -> IRModule:
        # warmup printer
        self.printer(func)

        self._program = func

        self.check_emitter_existence()

        self._builder = FunctionBuilder(
            name=func.name + "_kernel",
            kind="cuda_kernel" if is_nvgpu() else "hip_kernel",
            label="",
            grid_dim=self._program.num_blocks,
            block_dim=func.num_warps * 32,
            dynamic_smem_bytes=None,
            min_blocks=None,
        )
        self.builder.extend_params(func.params)

        # init for_thread_group stack
        self.thread_groups.num_groups = [1]
        self.thread_groups.group_size = [func.num_warps * 32]
        self.thread_groups.current_worker = [threadIdx.x]

        # init pre-defined variables
        self.init_smem_workspace(func)

        # emit body
        self.visit(func.body)

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

    def visit_SeqStmt(self, stmt: SeqStmt) -> None:
        for sub_stmt in stmt.seq:
            self.visit(sub_stmt)

    def visit_IfStmt(self, stmt: IfStmt) -> None:
        with self.builder.if_then(stmt.cond):
            self.visit(stmt.then_body)
        if stmt.else_body is not None:
            with self.builder.otherwise():
                self.visit(stmt.else_body)

    def visit_WhileStmt(self, stmt: WhileStmt) -> None:
        with self.builder.while_loop(stmt.cond):
            self.visit(stmt.body)

    def visit_ForStmt(self, stmt: ForStmt) -> None:
        if stmt.unroll_factor is None:
            attr = "."
        elif stmt.unroll_factor == -1:
            attr = "u"
        else:
            attr = "u{}".format(stmt.unroll_factor)  # no unroll
        with self.builder.for_loop(stmt.iter_var, stmt.extent, attr=attr):
            self.visit(stmt.body)

    def visit_ForThreadGroupStmt(self, stmt: ForThreadGroupStmt) -> None:
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

    def visit_BreakStmt(self, stmt: BreakStmt) -> None:
        self.builder.brk()

    def visit_DeclareStmt(self, stmt: DeclareStmt) -> None:
        self.builder.declare(stmt.var, init=stmt.init)

    def visit_AssignStmt(self, stmt: AssignStmt) -> None:
        self.builder.assign(stmt.var, value=stmt.value)

    def visit_TensorPtrStmt(self, stmt: TensorPtrStmt) -> None:
        self.builder.declare(stmt.ptr_var, self.tensor2var[stmt.tensor])

    def visit_InstStmt(self, stmt: InstStmt) -> None:
        self.visit(stmt.inst)

    def visit_Instruction(self, inst: Instruction) -> None:
        # insert a comment statement
        skip_comment_instructions = (PrintTensorInst, FormatPrintInst)
        if not isinstance(inst, skip_comment_instructions):
            self.builder.comment(str(self.printer(inst)), style="/*")

        # implement the vm instruction
        emitter_cls = resolve_inst_emitter(inst.__class__)
        if emitter_cls is None:
            raise RuntimeError("Can not resolve the emitter for instruction: {}".format(inst.__class__.__name__))
        emitter = emitter_cls(self)
        emitter.emit(inst)
        self.builder.append(emitter.finish())


class ProgramCodegen(IRFunctor):
    def __call__(self, prog: Program) -> IRModule:
        return self.visit(prog)

    def visit_Program(self, prog: Program) -> IRModule:
        ir_module = IRModule()
        for name, func in prog.functions.items():
            func_codegen = Codegen()
            sub_ir_module = func_codegen(func)
            ir_module = merge_ir_modules([ir_module, sub_ir_module])

        # if there is only one public function, we copy it and generate a function named 'launch', which is used as the
        # entry point of the module
        public_functions = [func for func in ir_module.functions.values() if func.kind == "public"]

        if len(public_functions) == 1 and "launch" not in ir_module.functions:
            public_func: HidetFunction = public_functions[0]
            ir_module.add_function(
                name="launch",
                func=HidetFunction(
                    name="launch",
                    params=public_func.params,
                    body=public_func.body,
                    ret_type=public_func.ret_type,
                    kind=public_func.kind,
                    attrs=public_func.attrs,
                ),
            )
        return ir_module


def generate_ir_module(prog: Program) -> IRModule:
    """
    Generate an IRModule from a Program by compiling the statements and instructions to lower-level Hidet IR.

    Parameters
    ----------
    prog: Program
        The program to be compiled.

    Returns
    -------
    ir_module: IRModule
        The lower-level Hidet IR module.
    """
    codegen = ProgramCodegen()
    ir_module: IRModule = codegen(prog)
    return ir_module
