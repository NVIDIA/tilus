# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import dataclasses
import functools
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import filelock
from hidet.backend.build import compile_source
from hidet.drivers.build_module import write_function_types
from hidet.ir.module import IRModule

import tilus.option
from tilus.backends.codegen import generate_ir_module
from tilus.extensions.hidet.backend.codegen import codegen
from tilus.ir.prog import Program
from tilus.ir.tools import verify
from tilus.runtime import CompiledProgram, compiled_program_exists, load_compiled_program


@dataclass(frozen=True)
class BuildOptions:
    """
    Options for building a program.
    """

    debug_block: Optional[tuple[int, int, int]] = None

    @staticmethod
    def _normalize_debug_block(debug_block: Optional[Sequence[int] | int]) -> Optional[tuple[int, int, int]]:
        if debug_block is None:
            return None
        else:
            if isinstance(debug_block, int):
                debug_block = [debug_block]
            debug_block = list(debug_block)
            while len(debug_block) < 3:
                debug_block.append(0)
            return debug_block[0], debug_block[1], debug_block[2]

    @staticmethod
    def create(debug_block: Optional[Sequence[int] | int] = None) -> BuildOptions:
        return BuildOptions(debug_block=BuildOptions._normalize_debug_block(debug_block))

    def with_debug_block(self, debug_block_: Optional[Sequence[int] | int] = None) -> BuildOptions:
        return dataclasses.replace(self, debug_block=BuildOptions._normalize_debug_block(debug_block_))


def optimize_program(program: Program, options: BuildOptions, cache_dir: Path) -> Program:
    """
    Optimize the program with a predefined set of transformations.

    Parameters
    ----------
    program: Program
        The program to optimize.

    options: BuildOptions
        The options for building the program.

    cache_dir: Path, optional
        The directory to store the cache of the current program. Used to store the IR when debug.dump_ir is set to True.

    Returns
    -------
    optimized_prog: Program
        The optimized program.
    """
    from tilus.transforms import Pass, PassContext, apply_transforms, get_default_passes, inject_print_instruction_pass

    transforms: list[Pass] = get_default_passes()

    if options.debug_block is not None:
        transforms.append(inject_print_instruction_pass(block_to_print=options.debug_block))

    with PassContext() as ctx:
        if tilus.option.get_option("debug.dump_ir"):  # dump the IR after each transformation
            ctx.dump_ir(cache_dir / "ir")

        return apply_transforms(program, transforms)


def optimize_ir_module(ir_module: IRModule, cache_dir: Path) -> IRModule:
    """
    Optimize the low-level IR module with a predefined set of transformations.

    Parameters
    ----------
    ir_module: IRModule
        The low-level IR module to optimize.

    cache_dir: Path
        The directory to store the cache of the current program. Used to store the IR when debug.dump_ir is set to True.

    Returns
    -------
    optimized_ir_module: IRModule
        The optimized low-level IR module.
    """
    from hidet.transforms import lower_with
    from hidet.transforms.add_explicit_cast import add_explicit_cast_pass
    from hidet.transforms.add_hints import add_hints_pass
    from hidet.transforms.annotate_header_and_libs import annotate_header_and_libs_pass
    from hidet.transforms.base import PassContext
    from hidet.transforms.check_launch_configuration import check_launch_configuration_pass
    from hidet.transforms.expand_let_expr import expand_let_expr_pass
    from hidet.transforms.explicit_unroll import explicit_unroll_pass
    from hidet.transforms.flatten_tensor_index import flatten_tensor_index_pass
    from hidet.transforms.flatten_tensor_slice import flatten_tensor_slice_pass
    from hidet.transforms.generate_launch_func import generate_launch_func_pass
    from hidet.transforms.import_primitive_functions import import_primitive_functions_pass
    from hidet.transforms.inline_function import inline_function_pass
    from hidet.transforms.inline_let_stmt import inline_let_stmt_pass
    from hidet.transforms.instantiate_symbols import instantiate_symbols_pass
    from hidet.transforms.instruments import PassInstrument, ProfileInstrument, SaveIRInstrument
    from hidet.transforms.lower_integer_subbyte import lower_integer_subbyte_pass
    from hidet.transforms.lower_special_cast import lower_special_cast_pass
    from hidet.transforms.lower_task_mapping import lower_task_mapping_pass
    from hidet.transforms.propagate_launch_bound import propagate_launch_bound_pass
    from hidet.transforms.resolve_generic_primitive_function import resolve_primitive_func_pass
    from hidet.transforms.simplify_addition_chain import simplify_addition_chain_pass
    from hidet.transforms.simplify_stmt import simplify_stmt_pass
    from hidet.transforms.unify_global_objects import unify_global_objects_pass

    from tilus.backends.transforms.inline_register_tensor import inline_register_tensor_pass
    from tilus.extensions.hidet.transforms.add_explicit_cast import (
        add_explicit_cast_pass as tilus_add_explicit_cast_pass,
    )
    from tilus.extensions.hidet.transforms.deadcode_elimination import deadcode_elimination_pass
    from tilus.extensions.hidet.transforms.declare_to_let import declare_to_let_pass
    from tilus.extensions.hidet.transforms.hoist_loop_invariants import hoist_loop_invariants_pass
    from tilus.extensions.hidet.transforms.lower_affine_to_recurence import lower_affine_to_recurrence_pass
    from tilus.extensions.hidet.transforms.lower_subbyte_type import lower_subbyte_type_pass
    from tilus.extensions.hidet.transforms.rule_based_simplifier import rule_based_simplify_pass

    transforms = [
        unify_global_objects_pass(),
        inline_register_tensor_pass(),
        resolve_primitive_func_pass(),
        tilus_add_explicit_cast_pass(),
        lower_subbyte_type_pass(),
        generate_launch_func_pass(),
        propagate_launch_bound_pass(),
        flatten_tensor_slice_pass(),
        flatten_tensor_index_pass(),
        declare_to_let_pass(),
        lower_task_mapping_pass(),
        rule_based_simplify_pass(),  # make ir more readable
        lower_special_cast_pass(),
        inline_function_pass(),
        resolve_primitive_func_pass(),
        import_primitive_functions_pass(),
        resolve_primitive_func_pass(),
        import_primitive_functions_pass(),
        lower_integer_subbyte_pass(),
        add_explicit_cast_pass(),
        declare_to_let_pass(),
        instantiate_symbols_pass(),
        import_primitive_functions_pass(),
        check_launch_configuration_pass(),
        # simplification
        expand_let_expr_pass(),
        inline_let_stmt_pass(),
        explicit_unroll_pass(),
        rule_based_simplify_pass(),
        simplify_addition_chain_pass(),
        lower_affine_to_recurrence_pass(),
        hoist_loop_invariants_pass(),
        add_hints_pass(),
        inline_let_stmt_pass(),
        simplify_stmt_pass(),
        deadcode_elimination_pass(),
        annotate_header_and_libs_pass(),
    ]

    instruments: list[PassInstrument] = []
    if tilus.option.get_option("debug.dump_ir"):
        instruments.append(SaveIRInstrument(str(cache_dir / "module" / "ir")))
        instruments.append(ProfileInstrument(str(cache_dir / "module" / "ir" / "lower_time.txt")))

    with PassContext(instruments):
        return lower_with(ir_module, transforms)


@functools.lru_cache(maxsize=1024)
def get_cache_dir(prog: Program, options: BuildOptions) -> Path:
    """
    Resolve the cache directory for the program.

    It will first check if the program has been cached. If not, it will create a new cache directory and write the
    program text to the file. The cache directory is determined by the SHA256 hash of the program text.

    Parameters
    ----------
    prog: Program
        The program to determine the cache directory.

    options: BuildOptions
        The options for building the program.

    Returns
    -------
    cache_dir: Path
        The cache directory.
    """
    prog_text: str = str(prog)
    options_text: str = str(options)
    hex_digest: str = hashlib.sha256(options_text.encode() + prog_text.encode()).hexdigest()[:12]
    cache_dir: Path = Path(tilus.option.get_option("cache_dir")) / "programs" / hex_digest
    program_path: Path = cache_dir / "program.txt"
    options_path: Path = cache_dir / "options.txt"

    if program_path.exists() and options_path.exists():
        # make sure the program is the same as the cached one
        with open(program_path, "r") as f:
            cached_prog_text = f.read()
        with open(options_path, "r") as f:
            cached_options_text = f.read()
        if cached_prog_text != prog_text or cached_options_text != options_text:
            raise ValueError("The program text is different from the cached one: {}".format(program_path))
    else:
        # create the cache directory and write the program text to the file
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(program_path, "w") as f:
            f.write(prog_text)
        with open(options_path, "w") as f:
            f.write(options_text)

    return cache_dir


def build_program(
    prog: Program, options: Optional[BuildOptions] = None, load: bool = True
) -> Optional[CompiledProgram]:
    """
    Build the program into a compiled program that could be executed directly.

    Parameters
    ----------
    prog: Program
        The program to build.

    options: BuildOptions, optional
        The options for building the program.

    load: bool
        Whether to load the compiled module after building. Default is False.

    Returns
    -------
    compiled_module: CompiledProgram, optional
        The compiled program.
    """
    if options is None:
        options = BuildOptions()
    cache_dir: Path = get_cache_dir(prog, options)

    # the program has finished building the program, load the compiled module
    if compiled_program_exists(cache_dir):
        if load:
            return load_compiled_program(cache_dir)
        else:
            return None

    # lock the cache directory to prevent multiple processes from building the program at the same time
    lock_path = cache_dir / ".lock"
    with filelock.FileLock(str(lock_path)):
        # check if the program has been built by another process
        if compiled_program_exists(cache_dir):
            return load_compiled_program(cache_dir)

        # otherwise, build the program
        # 0. verify the program
        verify(prog)

        # 1. optimize the program
        prog = optimize_program(prog, options=options, cache_dir=cache_dir)

        # 2. generate the low-level IR (Hidet IR)
        ir_module: IRModule = generate_ir_module(prog)

        # 3. optimize the low-level IR
        ir_module = optimize_ir_module(ir_module, cache_dir)

        # 4. generate the low-level code (CUDA C)
        module_dir = cache_dir / "module"
        src_path = module_dir / "source.cu"
        codegen(ir_module, src_out_path=str(src_path), target="cuda")

        # 5. save the function types to func_types.pickle so that we know what functions are inside the lib.so
        write_function_types(ir_module=ir_module, output_dir=str(module_dir))

        # 6. compile the low-level code
        lib_path = module_dir / "lib.so"
        compile_source(source_file=str(src_path), output_library_file=str(lib_path), target="cuda")

        if load:
            return load_compiled_program(cache_dir)
        else:
            return None
