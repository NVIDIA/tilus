from pathlib import Path
import hashlib
import filelock
from hidet.ir.module import IRModule
from hidet.runtime.compiled_module import CompiledModule, load_compiled_module, compiled_module_exists
from hidet.drivers.build_module import write_function_types
from hidet.backend.codegen import codegen
from hidet.backend.build import compile_source

import tilus.option
from tilus.ir.tools.printer import IRPrinter
from tilus.backends.codegen import generate_ir_module
from tilus.ir.prog import Program


def optimize_program(program: Program, cache_dir: Path) -> Program:
    """
    Optimize the program with a predefined set of transformations.

    Parameters
    ----------
    program: Program
        The program to optimize.

    cache_dir: Path, optional
        The directory to store the cache of the current program. Used to store the IR when debug.dump_ir is set to True.

    Returns
    -------
    optimized_prog: Program
        The optimized program.
    """
    from tilus.transforms import PassContext, apply_transforms
    from tilus.transforms import bound_aware_simplify_pass

    transforms = [
        bound_aware_simplify_pass(),
    ]

    with PassContext() as ctx:
        if tilus.option.get_option("debug.dump_ir"):  # dump the IR after each transformation
            ctx.dump_ir(cache_dir / "ir")

        return apply_transforms(program, transforms)


def optimize_ir_module(ir_module: IRModule) -> IRModule:
    """
    Optimize the low-level IR module with a predefined set of transformations.

    Parameters
    ----------
    ir_module: IRModule
        The low-level IR module to optimize.

    Returns
    -------
    optimized_ir_module: IRModule
        The optimized low-level IR module.
    """
    from hidet.transforms import lower

    return lower(ir_module)


def _resolve_cache_dir(prog: Program) -> Path:
    """
    Resolve the cache directory for the program.

    It will first check if the program has been cached. If not, it will create a new cache directory and write the
    program text to the file. The cache directory is determined by the SHA256 hash of the program text.

    Parameters
    ----------
    prog: Program
        The program to determine the cache directory.

    Returns
    -------
    cache_dir: Path
        The cache directory.
    """
    printer = IRPrinter()
    prog_text: str = str(printer(prog))
    hex_digest: str = hashlib.sha256(prog_text.encode()).hexdigest()[:12]
    cache_dir: Path = Path(tilus.option.get_option("cache_dir")) / hex_digest
    program_path: Path = cache_dir / "program.txt"

    if program_path.exists():
        # make sure the program is the same as the cached one
        with open(program_path, "r") as f:
            cached_prog_text = f.read()
        if cached_prog_text != prog_text:
            raise ValueError("The program text is different from the cached one: {}".format(program_path))
    else:
        # create the cache directory and write the program text to the file
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(program_path, "w") as f:
            f.write(prog_text)

    return cache_dir


def build_program(prog: Program) -> CompiledModule:
    """
    Build the program into a compiled module that could be executed directly.

    Parameters
    ----------
    prog: Program
        The program to build.

    Returns
    -------
    compiled_module: CompiledModule
        The compiled module.
    """
    cache_dir: Path = _resolve_cache_dir(prog)
    module_dir: Path = cache_dir / "module"

    # the program has finished building the program, load the compiled module
    if compiled_module_exists(str(module_dir)):
        return load_compiled_module(str(module_dir))

    # lock the cache directory to prevent multiple processes from building the program at the same time
    lock_path = cache_dir / ".lock"
    with filelock.FileLock(str(lock_path)):
        # check if the program has been built by another process
        if compiled_module_exists(str(module_dir)):
            return load_compiled_module(str(module_dir))

        # otherwise, build the program
        # 1. optimize the program
        prog = optimize_program(prog, cache_dir)

        # 2. generate the low-level IR (Hidet IR)
        ir_module: IRModule = generate_ir_module(prog)

        # 3. optimize the low-level IR
        ir_module = optimize_ir_module(ir_module)

        # 4. generate the low-level code (CUDA C)
        src_path = module_dir / "source.cu"
        codegen(ir_module, src_out_path=str(src_path), target="cuda")

        # 5. save the function types to func_types.pickle so that we know what functions are inside the lib.so
        write_function_types(ir_module=ir_module, output_dir=str(module_dir))

        # 6. compile the low-level code
        lib_path = module_dir / "lib.so"
        compile_source(source_file=str(src_path), output_library_file=str(lib_path), target="cuda")

        return load_compiled_module(str(module_dir))
