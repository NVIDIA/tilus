# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from subprocess import PIPE
from typing import List, Optional, Sequence, cast

import torch
import tvm_ffi
import tvm_ffi.libinfo

import tilus.option
from tilus.hidet import libinfo
from tilus.hidet.libinfo import get_include_dirs
from tilus.target import Target


class CompilationFailed(Exception):
    def __init__(self, source_path: str, msg: str):
        super().__init__(source_path, msg)
        self.source_path = source_path
        self.msg = msg

    def __str__(self):
        lines = ["failed to compile file://{}".format(self.source_path), "{}".format(self.msg)]
        return "\n".join(lines)


class SourceCompiler:
    """The base class of source compiler."""

    def compile(
        self,
        src_path: str,
        out_lib_path: str,
        target: Target,
        include_dirs: Sequence[str] = (),
        linking_dirs: Sequence[str] = (),
        linking_libs: Sequence[str] = (),
        object_files: Sequence[str] = (),
    ) -> None:
        raise NotImplementedError()

    def run_compile_command(self, command: str, src_path: str, out_lib_path: str, keep_files: Sequence[str]) -> None:
        try:
            # the directory to store the library "lib.so"
            out_lib_dir = os.path.dirname(out_lib_path)

            # write the compilation command to "compile.sh"
            with open(os.path.join(out_lib_dir, "compile.sh"), "w") as f:
                f.write("#!/bin/bash\n\n")
                f.write(command)
                f.write("\n")

            # run the compilation command
            with tempfile.TemporaryDirectory() as working_dir:
                t1 = time.time()
                env = os.environ.copy()
                env["TMPDIR"] = working_dir
                result = subprocess.run(
                    command.split(), stderr=PIPE, stdout=PIPE, cwd=working_dir, env=env, check=False
                )
                t2 = time.time()

                for keep_file in keep_files:
                    file_path = os.path.join(working_dir, keep_file)
                    if os.path.exists(file_path):
                        shutil.copy(file_path, os.path.join(out_lib_dir, keep_file))

                # if the compilation failed, raise an exception
                if result.returncode:
                    message = "Command: {}\n".format(command)
                    if result.stdout:
                        message += result.stdout.decode().strip() + "\n"
                    if result.stderr:
                        message += result.stderr.decode().strip()
                    raise CompilationFailed(src_path, message)

                # write the compilation log
                log_name = self.__class__.__name__.lower() + "_output.txt"
                with open(os.path.join(out_lib_dir, log_name), "w", encoding="utf-8") as f:
                    output = "\n".join([result.stdout.decode("utf-8").strip(), result.stderr.decode("utf-8").strip()])
                    f.write(output.strip())
                    f.write("\n")
                    f.write("elapsed time: {:.3f} seconds".format(t2 - t1))

                    lines = output.split("\n")
                    warning_lines = [line for line in lines if "warning" in line]
                    warning_lines = warning_lines[: len(warning_lines) // 2]  # nvcc would print the same warning twice
                    if len(warning_lines) > 0:
                        warnings.warn("Compilation warnings:\n" + "\n".join(warning_lines))

        except subprocess.CalledProcessError as e:
            print(command)
            print(e.stderr.decode("utf-8"))
            raise e


class NVCC(SourceCompiler):
    def __init__(self):
        super().__init__()
        self.nvcc_path: str = cast(str, self._resolve_tool("nvcc", required=True))
        self.cuobjdump_path: Optional[str] = self._resolve_tool("cuobjdump")
        self.nvdisasm_path: Optional[str] = self._resolve_tool("nvdisasm")
        self.include_dirs: List[str] = get_include_dirs()
        tvm_ffi_lib = tvm_ffi.libinfo.find_libtvm_ffi()
        self.library_dirs: List[str] = [os.path.dirname(tvm_ffi_lib)] if tvm_ffi_lib else []

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _resolve_tool(name: str, required: bool = False) -> Optional[str]:
        path: Optional[str] = shutil.which(name)
        if path is not None:
            return path
        for try_dir in ["/usr/local/cuda/bin/", "/usr/bin"]:
            path = os.path.join(try_dir, name)
            if os.path.exists(path):
                return path
        if required:
            raise FileNotFoundError("Can not find {}.".format(name))
        return None

    def compile(
        self,
        src_path: str,
        out_lib_path: str,
        target: Target,
        include_dirs: Sequence[str] = (),
        linking_dirs: Sequence[str] = (),
        linking_libs: Sequence[str] = (),
        object_files: Sequence[str] = (),
    ) -> None:
        if len(object_files) > 0 and out_lib_path.endswith(".o"):
            raise ValueError("Can not compile multiple objects into a single object file.")

        arch = "sm_{major}{minor}{suffix}".format(
            major=target.properties.compute_capability[0],
            minor=target.properties.compute_capability[1],
            suffix=target.properties.feature_suffix if target.properties.feature_suffix is not None else "",
        )
        cpu_arch = "native"

        # add tvm-ffi header paths
        include_dirs = list(include_dirs)
        include_dirs.append(tvm_ffi.libinfo.find_include_path())
        include_dirs.append(libinfo.find_include_path())

        # The following command compiles the cuda source code to a shared library
        # See https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
        # for more information about nvcc compilation.
        command = [
            # the path to nvcc compiler
            self.nvcc_path,
            "--keep" if tilus.option.get_option("debug.dump_ir") else "",
            # the included directories.
            *["-I{}".format(include_dir) for include_dir in self.include_dirs + list(include_dirs)],
            # the library directories.
            *["-L{}".format(library_dir) for library_dir in self.library_dirs + list(linking_dirs)],
            *["-l{}".format(library) for library in [*linking_libs, "cuda"]],
            # optimize host side code via -O3
            "-O3",
            # host compiler options: enable openmp, avx2, unroll loops and fast math
            "-Xcompiler -fPIC,-m64,-march={cpu_arch},-O3,-funroll-loops,-ffast-math".format(cpu_arch=cpu_arch),
            # use c++11 standard
            "-std=c++17",
            # the target PTX and SASS version.
            "-gencode arch=compute_{cc},code=sm_{cc}".format(cc=arch[len("sm_") :]),
            # allow ptxas (PTX assembler) to output information like register/smem usage.
            "--ptxas-options=-v" + (",--opt-level=0" if tilus.option.get_option("debug.disable_ptxas_opt") else ""),
            # compile into position independent code.
            # '--compiler-options -fPIC,-m64,-mavx2,-march=native, -O3',
            # embed the line information into the binary, allow Nsight Compute to get the source code for profiling.
            "-lineinfo",
            # ftz=true and prec-div=false for fast math
            "-ftz={}".format("true" if True else "false"),
            "-prec-div={}".format("true" if False else "false"),
            # link tvm_ffi runtime (provides TVMFFIEnvGetStream and other C env APIs)
            "-ltvm_ffi",
            # shared cuda runtime library is used (.so), instead of static one (.a). used to reduce binary size.
            "--cudart shared",
            # allow constexpr function to be called from device code.
            # '--expt-relaxed-constexpr',
            # supress some warnings
            # see https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#generic-tool-options-diag-suppress
            # supress warming no 177 like: "warning #177-D: variable "xxx" was declared but never referenced"
            "--diag-suppress 177",
            # supress warning no 179 like: "warning #179-D: right operand of "%" is zero"
            "--diag-suppress 179",
            # supress warning no 39 like: "warning #39-D: division by zero"
            "--diag-suppress 39",
            # supress warning no 550 like: "warning #550-D: variable "xxx" was set but never used"
            "--diag-suppress 550",
            # generate shared library (lib.so).
            "--shared" if out_lib_path.endswith(".so") else "--compile",
            # the linking objects.
            " ".join(object_files),
            # the source path.
            src_path,
            # the output library path.
            "-o",
            out_lib_path,
        ]

        keep_files = []
        if tilus.option.get_option("debug.dump_ir"):
            keep_files.append("source.ptx")

        self.run_compile_command(" ".join(command), src_path, out_lib_path, keep_files)

        if tilus.option.get_option("debug.dump_ir"):
            # dump SASS with source line info using nvdisasm -g on extracted cubins
            target_sass_path = src_path.removesuffix(".cu") + ".sass"
            if self.cuobjdump_path is not None:
                with tempfile.TemporaryDirectory() as extract_dir:
                    try:
                        # extract cubin(s) from the shared library
                        subprocess.run(
                            [self.cuobjdump_path, "-xelf", "all", out_lib_path],
                            cwd=extract_dir,
                            stdout=PIPE,
                            stderr=PIPE,
                            check=True,
                        )
                        # find all extracted cubin files
                        cubin_files = sorted(
                            os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith(".cubin")
                        )
                        with open(target_sass_path, "w", encoding="utf-8") as f:
                            for cubin_file in cubin_files:
                                result = subprocess.run(
                                    [self.nvdisasm_path, "-g", "-c", cubin_file],
                                    stdout=PIPE,
                                    stderr=PIPE,
                                    check=True,
                                )
                                f.write(result.stdout.decode("utf-8"))
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        with open(target_sass_path, "w", encoding="utf-8") as f:
                            f.write("Failed to dump SASS code from {}: {}\n".format(out_lib_path, str(e)))


def compile_source(
    source_file: str,
    output_library_file: str,
    target: Target,
    include_dirs: Sequence[str] = (),
    linking_dirs: Sequence[str] = (),
    linking_libraries: Sequence[str] = (),
    object_files: Sequence[str] = (),
) -> None:
    """
    Compile the source code in 'src_path' file and output the library to 'out_lib_path'.

    Parameters
    ----------
    source_file: str
        The path to source code.
    output_library_file: str
        The path to output library.
    target: str or Target
        The target platform. Currently only support 'cpu' and 'gpu'.
    include_dirs: Sequence[str]
        The include directories.
    linking_dirs: Sequence[str]
        The library directories.
    linking_libraries:
        The libraries to link to the output library.
    object_files: Sequence[str]
        The path to object files. If not None, the object files will be linked to the output library.
    """
    source_file = os.path.abspath(source_file)
    output_library_file = os.path.abspath(output_library_file)
    if object_files is not None:
        object_files = [os.path.abspath(object_file) for object_file in object_files]

    if target.is_nvgpu():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        compiler = NVCC()
    else:
        raise ValueError("Unknown target platform: {}".format(target))

    object_files = object_files or []
    compiler.compile(
        source_file,
        output_library_file,
        target,
        include_dirs=include_dirs,
        linking_dirs=linking_dirs,
        linking_libs=linking_libraries,
        object_files=object_files,
    )
