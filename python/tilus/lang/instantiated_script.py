# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import hashlib
import inspect
import json
import logging
import os
import pickle
import shutil
import tempfile
import traceback
from functools import lru_cache
from itertools import product
from operator import itemgetter
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Type

import filelock
import tabulate
import torch
import tvm_ffi
from tqdm import tqdm

import tilus.option
from tilus.compiler_fingerprint import backend_fingerprint, frontend_fingerprint
from tilus.drivers import BuildOptions, build_program, get_cache_dir
from tilus.hidet.ir.type import DataType, PointerType, TensorPointerType
from tilus.hidet.utils.py import nocolor
from tilus.ir.prog import Program
from tilus.lang.script import Script
from tilus.lang.script_dependencies import script_dependency_fingerprint
from tilus.runtime import CompiledProgram, compiled_program_exists, load_compiled_program
from tilus.target import get_current_target, lazy_init
from tilus.utils import benchmark_func, relative_to_with_walk_up, to_snake_case
from tilus.utils.multiprocess import parallel_imap

logger = logging.getLogger(__name__)


def span_space(space: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """
    Span the space of autotune.

    Examples
    --------
    > span_space({'m': [1, 2, 3], 'n, k': [[1, 2], [3, 4]]})
    [
        {'m': 1, 'n': 1, 'k': 2},
        {'m': 1, 'n': 3, 'k': 4},
        {'m': 2, 'n': 1, 'k': 2},
        {'m': 2, 'n': 3, 'k': 4},
        {'m': 3, 'n': 1, 'k': 2},
        {'m': 3, 'n': 3, 'k': 4},
    ]

    Parameters
    ----------
    space: Mapping[str, Sequence[Any]]
        The space of autotune. The key in the space may contain multiple names in the Script's parameters.

    Returns
    -------
    spanned_space: list[dict[str, Any]]
        The spanned space of autotune.
    """
    keys = []
    values = []

    for key, value in space.items():
        subkeys: list[str] | str
        if "," in key:
            subkeys = key.split(", ")
        else:
            subkeys = key
        keys.append(subkeys)
        values.append(value)

    spanned_space = []
    for combination in product(*values):
        spanned_dict = {}
        for subkeys, comb in zip(keys, combination):
            if isinstance(subkeys, list):
                if not isinstance(comb, list) and len(subkeys) != len(comb):
                    raise ValueError("The length of the subkeys should be the same as the length of the combination.")
                for subkey, val in zip(subkeys, comb):
                    spanned_dict[subkey] = val
            else:
                spanned_dict[subkeys] = comb
        spanned_space.append(spanned_dict)

    return spanned_space


def generate_schedules(
    space: Mapping[str, Sequence[Any]],
    script_cls: Type[Script],
    script_args: Sequence[Any],
    script_kwargs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Generate schedules from the space of autotune and the user's given arguments to the Script class.

    Parameters
    ----------
    space: Mapping[str, Sequence[Any]]
        The space of autotune from @autotune.
    script_cls: Type[Script]
        The Script class to generate schedules.
    script_args: Sequence[Any]
        The arguments of the Script class from user.
    script_kwargs: Mapping[str, Any]
        The keyword arguments of the Script class from user.

    Returns
    -------
    schedules: list[dict[str, Any]]
        The generated schedules.
    """
    schedules = []
    init_func = getattr(script_cls, "__init__")
    init_args = [None] + list(script_args)  # the first argument is used to bind the 'self' parameter
    signature = inspect.signature(init_func)

    if script_cls.debug_schedule:
        spanned_space = [script_cls.debug_schedule]
    else:
        spanned_space = span_space(space)

    for spanned_dict in spanned_space:
        conflict_names = set(spanned_dict) & set(script_kwargs)
        if conflict_names:
            raise ValueError(
                "The autotune space tries to overwrite the default values of the Script __init__ function: {}".format(
                    conflict_names
                )
            )
        init_kwargs = dict(script_kwargs) | spanned_dict
        try:
            bound_args = signature.bind(*init_args, **init_kwargs)
            bound_args.apply_defaults()
        except TypeError as e:
            stack = inspect.stack()
            frame = stack[4]
            code_context = "" if frame.code_context is None else frame.code_context[0].strip()
            raise TypeError(
                str(e) + f'\n  File "{frame.filename}", line {frame.lineno}, in {frame.function}\n    {code_context}'
            ) from None
        schedule = dict(bound_args.arguments)
        schedule.pop("self")  # remove the 'self' argument
        schedules.append(schedule)
    return schedules


def _safe_str(fn: Any) -> str:
    """Call ``fn`` and stringify its result, returning ``"unknown"`` on any failure."""
    try:
        return str(fn())
    except Exception:
        return "unknown"


def _tilus_version() -> str:
    """Release-level tilus version used in the cache fingerprint.

    We deliberately key on the *base* release (e.g. ``0.2.1``) rather than the full SCM version
    (``0.2.1.dev19+g<hash>``). The SCM version changes on every commit during development, which would
    needlessly invalidate the dispatch cache each time; keying on the base release lets all dev builds
    off the same release share the cache while distinct releases stay separated.
    """
    try:
        import importlib.metadata as importlib_metadata

        raw = importlib_metadata.version("tilus")
    except Exception:
        raw = str(getattr(tilus, "__version__", "unknown"))
    try:
        from packaging.version import Version

        return Version(raw).base_version
    except Exception:
        # strip a PEP 440 dev / local suffix by hand if packaging is unavailable
        return raw.split(".dev")[0].split("+")[0]


def collect_tuning_metadata() -> dict[str, str]:
    """Fingerprint of the environment that produced a tuning dispatch table.

    The fastest schedule for a tuning key depends on the GPU and the toolchain, so a dispatch table
    tuned in one environment must not be silently reused in another (for example, a table tuned on a
    B200 picked up on a B300 through a shared cache directory). These fields are written next to the
    dispatch table and compared on load; a mismatch causes the on-disk table to be ignored and
    re-tuned. This mirrors the metadata-validation approach used by FlashInfer's autotuner cache.
    """
    return {
        "tilus_version": _tilus_version(),
        "target": _safe_str(get_current_target),
        "gpu": _safe_str(lambda: torch.cuda.get_device_name(torch.cuda.current_device())),
        "compute_capability": _safe_str(lambda: "{}.{}".format(*torch.cuda.get_device_capability())),
        "cuda_version": _safe_str(lambda: torch.version.cuda),
    }


def tuning_metadata_matches(saved: Any, current: Mapping[str, str]) -> bool:
    """Return whether a saved metadata block is compatible with the current environment.

    Every field present in ``current`` must match the corresponding field in ``saved``. A saved field
    set to the wildcard ``"*"`` matches any value, allowing advanced users to relax individual checks
    by editing the cache file by hand (as in FlashInfer). A missing or non-mapping ``saved`` (e.g. a
    legacy cache file without metadata) never matches.
    """
    if not isinstance(saved, dict):
        return False
    for key, current_value in current.items():
        saved_value = saved.get(key)
        if saved_value == "*":
            continue
        if saved_value != current_value:
            return False
    return True


class CallParameters:
    """
    Analyze the parameters in the Script '__call__' function.

    We require that each parameter in the '__call__' function has type annotation. The type annotation be either
    - a pythonic constant, e.g., int, float, str, bool, etc.
    - a type from Hidet IR's type system, e.g., hidet.float16, hidet.int32, etc.

    The return annotation of the '__call__' function should always be None for now.

    We treat the pythonic constant parameters as the JIT constants, that is, different values of these parameters will
    trigger different JIT compilations.
    We treat the hidet integer types (hidet.int32, hidet.int64, etc.) as the parameters that will trigger tuning but
    not JIT compilation.
    """

    def __init__(self, script_cls: Type[Script]):
        self.signature: inspect.Signature = inspect.signature(getattr(script_cls, "__call__"))
        self.param_names: list[str] = []
        self.param_types: list[Type[bool] | Type[int] | Type[float] | Type[str] | DataType | PointerType] = []
        self.const_params: list[int] = []
        self.kernel_params: list[int] = []
        self.tuning_params: list[int] = []
        self.with_default: bool = False

        self.extract_params()

    def extract_params(self):
        for index, param in enumerate(list(self.signature.parameters.values())[1:]):
            # check that there is only 'POSITIONAL_OR_KEYWORD' kind of parameters
            # see https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise ValueError("The parameter kind should be POSITIONAL_OR_KEYWORD for each __call__ parameter.")

            # check that the parameter has type annotation
            if param.annotation is inspect.Parameter.empty:
                raise ValueError("The following parameter miss type annotation: " + param.name)

            # check if the parameter has default value
            if param.default is not inspect.Parameter.empty:
                self.with_default = True

            # check that the parameter type is either a pythonic constant or a Hidet IR type
            if isinstance(param.annotation, (DataType, PointerType, TensorPointerType)) or param.annotation in [
                bool,
                int,
                float,
                str,
            ]:
                self.param_names.append(param.name)
                self.param_types.append(param.annotation)

                annotation = param.annotation
                if annotation in [bool, int, float, str]:
                    self.const_params.append(index)
                else:
                    self.kernel_params.append(index)
                    if isinstance(annotation, DataType) and annotation.is_integer():
                        self.tuning_params.append(index)
            elif isinstance(param.annotation, str):
                # when `import __future__ import annotations` is used, all annotations will become strings
                # we do not support this feature for now, try asking the user avoid using this feature.
                raise ValueError(
                    "The parameter annotation is a string. "
                    "It's likely that `from __future__ import annotations` is used. "
                    "Tilus currently does not support this feature and please remove it and try again."
                )
            else:
                raise ValueError("The type annotation should be a pythonic constant or a Hidet IR type.")

        # check that the return annotation is None
        if self.signature.return_annotation not in [self.signature.empty, None]:
            raise ValueError("The return annotation of the __call__ function should be None or omitted.")


JitKey = tuple[Any, ...]
TuningKey = tuple[int, ...]
divisibility_key: tuple[int, ...]


def _init_divisibility_key():
    global divisibility_key
    divisibility_key_list = []
    multiples = [1]
    for n in range(32):
        for m in reversed(multiples):
            if n % m == 0:
                divisibility_key_list.append(m)
                break
        else:
            assert False
    divisibility_key = tuple(divisibility_key_list)


_init_divisibility_key()


@lru_cache(maxsize=1024)
def _tuning_key_part(arg: int) -> tuple[int, int]:
    """Return the divisibility and bucket for one tuning argument."""
    block = 1 << max((arg.bit_length() - 2), 0)
    return divisibility_key[arg % 32], (arg + block - 1) // block * block


def extract_keys(args: Sequence[Any], const_params: list[int], tuning_params: list[int]) -> tuple[JitKey, TuningKey]:
    """
    Extract the JIT key and the tuning key from the arguments.

    This function is on the hot path, thus we should try to maximize its performance.

    Parameters
    ----------
    args: Sequence[Any]
        The calling arguments to __call__ of Script.

    const_params: list[int]
        The index of the constant parameters in the __call__ function.

    tuning_params: list[int]
        The index of the tuning parameters in the __call__ function.

    Returns
    -------
    keys: tuple[JitKey, TuningKey]
        The JIT key and the tuning key.
    """
    jit_key = []
    tuning_key = []
    for i in const_params:
        jit_key.append(args[i])
    for i in tuning_params:
        arg: int = args[i]
        divisibility, bucket = _tuning_key_part(arg)
        jit_key.append(divisibility)
        tuning_key.append(bucket)
    return tuple(jit_key), tuple(tuning_key)


def construct_keys(const_params: Sequence[Any], tuning_params: Sequence[int]) -> tuple[JitKey, TuningKey]:
    jit_key = list(const_params)
    tuning_key = []
    for arg in tuning_params:
        divisibility, bucket = _tuning_key_part(arg)
        tuning_key.append(bucket)
        jit_key.append(divisibility)
    return tuple(jit_key), tuple(tuning_key)


def _make_tuple_getter(indices: Sequence[int]) -> Callable[[Sequence[Any]], tuple[Any, ...]]:
    """Build a low-overhead positional selector that always returns a tuple."""
    indices = tuple(indices)
    if not indices:
        return lambda args: ()
    if len(indices) == 1:
        index = indices[0]
        return lambda args: (args[index],)
    return itemgetter(*indices)


class JitInstance:
    SPECIALIZATION_CACHE_VERSION = 3

    def __init__(
        self,
        script_cls: Type[Script],
        call_params: CallParameters,
        build_options: BuildOptions,
        schedules: list[dict[str, Any]],
        jit_key: JitKey,
    ):
        self.script_cls: Type[Script] = script_cls
        self.instance_name: str = self._instance_name(script_cls, call_params, jit_key)
        self.call_params: CallParameters = call_params
        self.build_options: BuildOptions = build_options
        self.schedules: list[dict[str, Any]] = schedules
        self.jit_key: JitKey = jit_key

        # the programs that have been successfully transpiled
        self.transpiled_programs: list[Program] = []
        self.transpiled_schedules: list[int] = []
        self.failed_scheduling: list[str] = []

        # the programs that have been successfully transpiled and built
        self.valid_programs: list[Program] = []
        self.valid_schedules: list[int] = []
        self.compiled_programs: list[CompiledProgram] = []
        self.failed_building: list[tuple[str, str]] = []

        self.dispatch_table: dict[TuningKey, int] = {}
        self.cache_dir: Path = Path()
        self.cache_dir_lock: Path = Path()
        cache_root = str(tilus.option.get_option("cache_dir"))
        self.frontend_fingerprint = frontend_fingerprint(cache_root)
        self.backend_fingerprint = backend_fingerprint(cache_root)
        self.specialization_cache_path = self._get_specialization_cache_path()

        if not self._load_specialization_cache():
            self._transpile_programs()

    def __str__(self):
        data = {
            "Script": self.script_cls.__qualname__,
            "Cache Dir": str(self.cache_dir),
            "Build Options": str(self.build_options),
            "Num Schedules": len(self.schedules),
            "Valid Programs": max(len(self.transpiled_programs), len(self.compiled_programs)),
        }
        return str(tabulate.tabulate(data.items(), tablefmt="simple", colalign=("right", "left")))

    def programs(self) -> Sequence[Program]:
        if len(self.valid_programs) == 0:
            self._build_programs()
        return self.valid_programs

    def _get_specialization_cache_path(self) -> Path | None:
        param_info = self.call_params
        inputs = (
            self.schedules,
            self.jit_key,
            param_info.param_names,
            param_info.const_params,
            param_info.kernel_params,
            param_info.tuning_params,
        )
        dependencies = script_dependency_fingerprint(self.script_cls, inputs)
        if dependencies is None:
            return None
        cache_key = repr(
            (
                self.SPECIALIZATION_CACHE_VERSION,
                self.frontend_fingerprint,
                dependencies,
                self.build_options,
                [str(param_type) for param_type in param_info.param_types],
                get_current_target(),
            )
        )
        digest = hashlib.sha256(cache_key.encode()).hexdigest()[:20]
        return Path(tilus.option.get_option("cache_dir")) / "specializations" / f"{digest}.json"

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
            os.replace(temp_name, path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def _load_specialization_cache(self) -> bool:
        path = self.specialization_cache_path
        if path is None or not path.exists():
            return False
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if data.get("version") != self.SPECIALIZATION_CACHE_VERSION:
                return False
            if data.get("frontend_fingerprint") != self.frontend_fingerprint:
                return False

            cache_root = Path(tilus.option.get_option("cache_dir")).resolve()
            cache_dir = (cache_root / data["script_cache_dir"]).resolve()
            programs_path = (cache_root / data["programs_file"]).resolve()
            if (
                not cache_dir.is_relative_to(cache_root)
                or not programs_path.is_relative_to(cache_root)
                or not cache_dir.is_dir()
            ):
                return False

            programs_payload = programs_path.read_bytes()
            if hashlib.sha256(programs_payload).hexdigest() != data["programs_digest"]:
                return False
            transpiled_schedules, transpiled_programs = pickle.loads(programs_payload)
            if (
                not isinstance(transpiled_schedules, list)
                or not isinstance(transpiled_programs, list)
                or len(transpiled_schedules) != len(transpiled_programs)
                or not transpiled_programs
                or any(not isinstance(program, Program) for program in transpiled_programs)
                or any(
                    not isinstance(index, int) or index < 0 or index >= len(self.schedules)
                    for index in transpiled_schedules
                )
            ):
                return False
        except (
            AttributeError,
            EOFError,
            ImportError,
            KeyError,
            OSError,
            pickle.PickleError,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return False

        self.cache_dir = cache_dir
        self.cache_dir_lock = cache_dir / ".lock"
        self.transpiled_schedules = transpiled_schedules
        self.transpiled_programs = transpiled_programs
        self.valid_schedules = []
        self.valid_programs = []
        self.compiled_programs = []

        return True

    def _dump_specialization_cache(self) -> None:
        if self.specialization_cache_path is None:
            return
        cache_root = Path(tilus.option.get_option("cache_dir")).resolve()
        path = self.specialization_cache_path
        programs_payload = pickle.dumps(
            (self.transpiled_schedules, self.transpiled_programs),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        programs_digest = hashlib.sha256(programs_payload).hexdigest()
        programs_path = path.with_name(f"{path.stem}.{programs_digest[:16]}.programs.pkl")
        data = {
            "version": self.SPECIALIZATION_CACHE_VERSION,
            "frontend_fingerprint": self.frontend_fingerprint,
            "script_cache_dir": str(self.cache_dir.resolve().relative_to(cache_root)),
            "programs_file": str(programs_path.resolve().relative_to(cache_root)),
            "programs_digest": programs_digest,
        }
        self._atomic_write(programs_path, programs_payload)
        self._atomic_write(path, json.dumps(data, indent=2).encode())

    @staticmethod
    def _instance_name(script_cls: Type[Script], call_params: CallParameters, jit_key: JitKey) -> str:
        script_name = to_snake_case(script_cls.__name__)
        keys: list[Any] = list(jit_key)
        items = [script_name]
        for _ in call_params.const_params:
            items.append("{}".format(keys.pop(0)))

        for _ in call_params.tuning_params:
            items.append("d{}".format(keys.pop(0)))

        return "-".join(items)

    @staticmethod
    def _instantiate_schedule(job: Any) -> Program | str:
        from tilus.lang.transpiler import Transpiler

        assert len(job) == 5
        script_cls: Type[Script] = job[0]
        call_params: CallParameters = job[1]
        schedule: dict[str, Any] = job[2]
        name2const_py: dict[str, Any] = job[3]
        name2divisibility: dict[str, Any] = job[4]
        try:
            # we have redefined the __new__ for Script, thus we need to use object.__new__ to create the Script object
            script_obj = object.__new__(script_cls)

            # initialize the Script object
            script__init__ = getattr(script_cls, "__init__")
            script__init__(script_obj, **schedule)

            transpiler = Transpiler()
            name2consts: dict[str, Any] = {}
            for idx in call_params.const_params:
                name = call_params.param_names[idx]
                name2consts[name] = name2const_py[name]

            function = transpiler.transpile(script_obj, name2consts=name2consts, name2divisibility=name2divisibility)
            function = function.with_name(to_snake_case(script_cls.__name__))
            program = Program.create(functions={function.name: function})
            return program
        except Exception:
            return traceback.format_exc()

    @staticmethod
    def _build_program(job: Any) -> tuple[bool, str]:
        assert len(job) == 2
        program: Program = job[0]
        options: BuildOptions = job[1]
        try:
            program_cache_dir = build_program(program, options=options)
        except Exception:
            return False, traceback.format_exc()
        else:
            return True, program_cache_dir

    def _transpile_programs(self):
        # prepare the jobs for instantiating the schedules
        scheduling_jobs = []
        for idx, schedule in enumerate(self.schedules):
            # extract the constant parameters and the parameters with divisibility information from jit_key
            param_info = self.call_params
            num_constants = len(param_info.const_params)
            const_names = [param_info.param_names[i] for i in self.call_params.const_params]
            const_values = self.jit_key[:num_constants]
            tuning_names = [param_info.param_names[i] for i in self.call_params.tuning_params]
            tuning_values = self.jit_key[num_constants:]
            name2const = dict(zip(const_names, const_values))
            name2divisibility = dict(zip(tuning_names, tuning_values))
            scheduling_jobs.append((self.script_cls, self.call_params, schedule, name2const, name2divisibility))

        # instantiate the schedules into programs in parallel
        lazy_init()
        for idx, item in enumerate(
            tqdm(
                iterable=parallel_imap(func=JitInstance._instantiate_schedule, jobs=scheduling_jobs),
                desc="[{}] {}".format("Scheduling", self.instance_name),
                miniters=1,
                total=len(scheduling_jobs),
                ncols=60 + max(60, len(self.instance_name)),
                delay=3,
            )
        ):
            if isinstance(item, Program):
                self.transpiled_programs.append(item)
                self.transpiled_schedules.append(idx)
            else:
                assert isinstance(item, str)
                sections = {"Schedule": str(self.schedules[idx]), "Traceback": nocolor(item)}
                lines = []
                for key, value in sections.items():
                    lines.append(key)
                    lines.append("=" * len(key))
                    lines.append("")
                    lines.append(value)
                    lines.append("")
                self.failed_scheduling.append("\n".join(lines))

        # get the cache dir for this jit instance
        concatenated_program_text = "\n".join([str(program) for program in self.transpiled_programs])
        option_text = str(self.build_options)
        hash_string = option_text + concatenated_program_text
        hash_key = hashlib.sha256(hash_string.encode()).hexdigest()[:8]
        jit_name_items = [str(key) for key in self.jit_key] + [hash_key]
        self.cache_dir = (
            Path(tilus.option.get_option("cache_dir"))
            / "scripts"
            / to_snake_case(self.script_cls.__name__)
            / "-".join(jit_name_items)
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir_lock = self.cache_dir / ".lock"
        with filelock.FileLock(self.cache_dir_lock):
            # write the failed schedules to the cache dir
            failed_scheduling_dir = self.cache_dir / "failed" / "scheduling"
            if not failed_scheduling_dir.exists() and self.failed_scheduling or len(self.compiled_programs) == 0:
                shutil.rmtree(failed_scheduling_dir, ignore_errors=True)
                failed_scheduling_dir.mkdir(parents=True, exist_ok=True)
                for idx, failed_schedule in enumerate(self.failed_scheduling):
                    with open(failed_scheduling_dir / f"{idx}.txt", "w") as f:
                        f.write(failed_schedule)

            # write the source code of the script class to source.txt
            source_path = self.cache_dir / "source.txt"
            if not source_path.exists():
                try:
                    source_code = inspect.getsource(self.script_cls)
                except Exception:
                    # if the source code cannot be retrieved, we write an empty file
                    source_code = "Source code not available."
                with open(source_path, "w") as f:
                    f.write(source_code)

            # write the keys
            meta_path = self.cache_dir / "meta.json"
            if not meta_path.exists():
                names = self.call_params.param_names
                meta = {
                    "params": {
                        "names": names,
                        "types": [str(tp) for tp in self.call_params.param_types],
                        "const_params": [names[i] for i in self.call_params.const_params],
                        "kernel_params": [names[i] for i in self.call_params.kernel_params],
                        "tuning_params": [names[i] for i in self.call_params.tuning_params],
                    },
                    "jit_key": self.jit_key,
                    "jit_key_desc": "The const parameters and the divisibility of the tuning parameters, in order.",
                }
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=4)

            # write build options
            options_path = self.cache_dir / "build_options.txt"
            if not options_path.exists():
                with open(options_path, "w") as f:
                    f.write(str(self.build_options))

        if len(self.transpiled_programs) == 0:
            call_file = inspect.getsourcefile(self.script_cls.__call__)
            call_lino = inspect.getsourcelines(self.script_cls.__call__)[1]
            lines = [
                "No valid schedule found during instantiating the Tilus Script:",
                '  File "{}", line {}'.format(call_file, call_lino),
                "Please check the following cache dir for failure information:",
                "  {}".format(str(self.cache_dir.resolve())),
                "",
                "The first failed scheduling",
                "===========================",
                "",
            ]
            lines.extend(["  " + s for s in self.failed_scheduling[0].split("\n")])
            raise RuntimeError("\n".join(lines))

        self._dump_specialization_cache()

    @staticmethod
    def _create_link(link_path: Path, target_path: Path) -> None:
        relative_path = relative_to_with_walk_up(link_path.parent, target=target_path)
        try:
            os.symlink(relative_path, link_path)
        except FileExistsError:
            # Symlink already exists (e.g., from a previous run with a stale dispatch table).
            # Replace it to ensure it points to the correct target.
            os.remove(link_path)
            os.symlink(relative_path, link_path)

    def _build_programs(self):
        # build the programs in parallel
        transpiled_programs = self.transpiled_programs
        transpiled_schedules = self.transpiled_schedules
        # Resolve the ordinary binary key in the parent before starting workers.
        # A cache hit needs no compilation process, and every ambient build option
        # is still checked by get_cache_dir.
        cached_results = []
        building_jobs = []
        for program in transpiled_programs:
            program_cache_dir = get_cache_dir(program, options=self.build_options)
            if compiled_program_exists(program_cache_dir):
                cached_results.append((True, str(program_cache_dir)))
            else:
                cached_results.append(None)
                building_jobs.append((program, self.build_options))

        def build_results():
            pending = (
                iter(parallel_imap(func=JitInstance._build_program, jobs=building_jobs)) if building_jobs else iter(())
            )
            for cached in cached_results:
                yield cached if cached is not None else next(pending)

        lazy_init()
        for idx, (success, result) in enumerate(
            tqdm(
                iterable=build_results(),
                desc="[{}] {}".format("Building", self.instance_name),
                total=len(transpiled_programs),
                miniters=1,
                ncols=60 + max(60, len(self.instance_name)),
                delay=3,
            )
        ):
            if success:
                program_cache_dir = Path(result)
                self.valid_programs.append(transpiled_programs[idx])
                self.valid_schedules.append(transpiled_schedules[idx])
                self.compiled_programs.append(load_compiled_program(program_cache_dir))
            else:
                program_cache_dir = get_cache_dir(transpiled_programs[idx], options=self.build_options)
                sections = {
                    "Schedule": str(self.schedules[idx]),
                    "Program": str(transpiled_programs[idx]),
                    "Traceback": nocolor(result),
                }
                lines = []
                for key, value in sections.items():
                    lines.append(key)
                    lines.append("=" * len(key))
                    lines.append("")
                    lines.append(value)
                    lines.append("")
                self.failed_building.append(("\n".join(lines), str(program_cache_dir)))

        # some checks
        if self.script_cls.debug_block and len(self.compiled_programs) > 1:
            raise ValueError("Please specify the debug_schedule when debug_block is set. ")

        # initialize the script cache if it does not exist
        with filelock.FileLock(self.cache_dir_lock):
            # add soft link to the cache dirs of compiled programs
            programs_dir = self.cache_dir / "programs"
            if not programs_dir.exists():
                programs_dir.mkdir()
                for idx, compiled_program in enumerate(self.compiled_programs):
                    self._create_link(
                        link_path=(programs_dir / str(idx)).resolve(),
                        target_path=compiled_program.program_dir.resolve(),
                    )

            # write the schedule.txt
            schedule_path = self.cache_dir / "schedule.txt"
            if not schedule_path.exists() and self.valid_schedules:
                headers = ["index"] + list(self.schedules[self.valid_schedules[0]].keys())
                rows = []
                for idx in self.valid_schedules:
                    rows.append([idx] + list(self.schedules[idx].values()))
                with open(schedule_path, "w") as f:
                    f.write(tabulate.tabulate(rows, headers=headers))

            # write the failed building to the cache dir
            failed_building_dir = self.cache_dir / "failed" / "building"
            if not failed_building_dir.exists() and self.failed_building or len(self.compiled_programs) == 0:
                shutil.rmtree(failed_building_dir, ignore_errors=True)
                failed_building_dir.mkdir(parents=True, exist_ok=True)
                for idx, (failed_building, prog_cache_dir) in enumerate(self.failed_building):
                    with open(failed_building_dir / f"{idx}.txt", "w") as f:
                        f.write(failed_building)
                    self._create_link(
                        link_path=(failed_building_dir / f"{idx}").resolve(), target_path=Path(prog_cache_dir).resolve()
                    )

            # load the dispatch table from the cache dir
            self.load_dispatch_table()

        if len(self.compiled_programs) == 0:
            call_file = inspect.getsourcefile(self.script_cls.__call__)
            call_lino = inspect.getsourcelines(self.script_cls.__call__)[1]
            lines = [
                "No valid schedule found during building programs:",
                '  File "{}", line {}'.format(call_file, call_lino),
                "Please check the following cache dir for failure information:",
                "  {}".format(str(self.cache_dir.resolve())),
                "",
                "The first failed building",
                "=========================",
                "",
            ]
            lines.extend(["  " + s for s in self.failed_building[0][0].split("\n")])

            raise RuntimeError("\n".join(lines))

    def _pick_best_program(self, args: Sequence[Any]) -> CompiledProgram:
        if len(self.compiled_programs) == 0:
            self._build_programs()

        _, tuning_key = extract_keys(args, self.call_params.const_params, self.call_params.tuning_params)

        # check if the tuning key exists in the dispatch table
        if tuning_key not in self.dispatch_table:
            # load the dispatch table from the cache dir in case another process has updated it
            with filelock.FileLock(self.cache_dir_lock):
                self.load_dispatch_table()
                if tuning_key in self.dispatch_table:
                    return self.compiled_programs[self.dispatch_table[tuning_key]]

                # it's not in the dispatch table in memory nor in the cache dir
                latency: list[float] = []
                if len(self.compiled_programs) == 1:
                    # we skip the benchmark if there is only one compiled program
                    latency.append(0.0)
                else:
                    tuning_key_name = " " + "-".join([str(v) for v in tuning_key]) if tuning_key else ""
                    for i, compiled_program in tqdm(
                        iterable=enumerate(self.compiled_programs),
                        desc="[{}] {}{}".format("Tuning", self.instance_name, tuning_key_name),
                        miniters=1,
                        mininterval=0,
                        ncols=60 + max(60, len(self.instance_name) + len(tuning_key_name)),
                    ):
                        compiled_func = compiled_program.get_launch_func()
                        kernel_args = [
                            args[j].clone() if isinstance(args[j], torch.Tensor) else args[j]
                            for j in self.call_params.kernel_params
                        ]
                        try:
                            lat = benchmark_func(
                                lambda: compiled_func(*kernel_args),
                                warmup=tilus.option.get_option("bench_warmup"),
                                repeat=tilus.option.get_option("bench_repeat"),
                            )
                        except RuntimeError as e:
                            raise RuntimeError(
                                f"Failed to benchmark the kernel {self.instance_name} with schedule: \n"
                                f"  {str(self.schedules[self.valid_schedules[i]])}\n"
                                "Error message:\n"
                                f"  {str(e)}"
                            ) from e
                        latency.append(lat)  # type: ignore

                best_latency = min(latency)
                best_program_idx = latency.index(best_latency)
                self.dispatch_table[tuning_key] = best_program_idx
                self.dump_dispatch_table()

                # write the benchmark results, and link to the optimal program
                latency_dir = self.cache_dir / "latency" / "{}".format("-".join(str(k) for k in tuning_key))
                latency_dir.mkdir(parents=True, exist_ok=True)
                tuning_report_path = latency_dir / "report.txt"
                with open(tuning_report_path, "w") as f:
                    headers = ["index"] + list(self.schedules[self.valid_schedules[0]].keys()) + ["latency (ms)"]
                    rows = []
                    for i in range(len(self.valid_schedules)):
                        schedule_values = list(self.schedules[self.valid_schedules[i]].values())
                        rows.append([i] + schedule_values + [latency[i]])
                    rows = sorted(rows, key=lambda x: x[-1])
                    with open(tuning_report_path, "w") as f:
                        f.write(tabulate.tabulate(rows, headers=headers, floatfmt=".3f"))
                self._create_link(
                    link_path=latency_dir / str(best_program_idx),
                    target_path=self.compiled_programs[best_program_idx].program_dir,
                )

        return self.compiled_programs[self.dispatch_table[tuning_key]]

    def load_dispatch_table(self):
        table_path = self.cache_dir / "dispatch_table.json"
        if not table_path.exists():
            return
        with open(table_path, "r") as f:
            data = json.load(f)
        # The fastest schedule for a tuning key is environment-specific, so a dispatch table is only
        # reused when its saved environment fingerprint matches the current one. Otherwise (including a
        # legacy table that predates the fingerprint) it is ignored and the kernel is re-tuned.
        saved_meta = data.get("_metadata") if isinstance(data, dict) else None
        self.dispatch_table = {}
        if (
            not tuning_metadata_matches(saved_meta, collect_tuning_metadata())
            or data.get("_frontend_fingerprint") != self.frontend_fingerprint
            or data.get("_backend_fingerprint") != self.backend_fingerprint
        ):
            return
        self.dispatch_table = {
            tuple(key): value
            for key, value in data["entries"]
            if isinstance(value, int) and 0 <= value < len(self.compiled_programs)
        }

    def dump_dispatch_table(self):
        table_path = self.cache_dir / "dispatch_table.json"
        table_txt_path = self.cache_dir / "dispatch_table.txt"
        entries = [[list(key), value] for key, value in self.dispatch_table.items()]
        data = {
            "_metadata": collect_tuning_metadata(),
            "_frontend_fingerprint": self.frontend_fingerprint,
            "_backend_fingerprint": self.backend_fingerprint,
            "entries": entries,
        }
        with open(table_path, "w") as f:
            json.dump(data, f)
        headers = []
        for idx in self.call_params.tuning_params:
            headers.append(self.call_params.param_names[idx])
        headers.append("choice")
        rows = []
        for key, value in self.dispatch_table.items():
            row = list(key)
            row.append(value)
            rows.append(row)
        with open(table_txt_path, "w") as f:
            f.write(tabulate.tabulate(rows, headers=headers))


class InstantiatedScript:
    def __init__(self, script_cls: Type[Script], script_args: Sequence[Any], script_kwargs: Mapping[str, Any]):
        self.script_cls: Type[Script] = script_cls
        self.script_name: str = tilus.utils.to_snake_case(script_cls.__name__)
        self.space: Mapping[str, Sequence[Any]] = getattr(script_cls, "_autotune_space", {})
        self.build_options: BuildOptions = BuildOptions.create(debug_block=script_cls.debug_block)
        self.schedules: list[dict[str, Any]] = generate_schedules(self.space, script_cls, script_args, script_kwargs)

        self.params: CallParameters = CallParameters(script_cls)
        self.with_default: bool = self.params.with_default
        self.const_params: list[int] = self.params.const_params
        self.kernel_params: list[int] = self.params.kernel_params
        self.tuning_params: list[int] = self.params.tuning_params

        self.jit_instances: dict[JitKey, JitInstance] = {}
        self.dispatch_table: dict[tuple[JitKey, TuningKey], tvm_ffi.Function] = {}
        self._dispatch_arg_getter = _make_tuple_getter(self.const_params + self.tuning_params)
        self._kernel_arg_getter = _make_tuple_getter(self.kernel_params)
        self._last_dispatch_args: Optional[tuple[Any, ...]] = None
        self._last_compiled_func: Optional[tvm_ffi.Function] = None

    def __call__(self, *args, **kwargs):
        if kwargs or self.with_default:
            # we allow the user to pass the keyword arguments to the script instance, or use the default values
            bound_args = self.params.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            args = bound_args.args
        else:
            if len(args) != len(self.params.param_names):
                raise ValueError(
                    "The number of arguments should be {}, but got {}.".format(len(self.params.param_names), len(args))
                )

        # Most workloads launch the same shape repeatedly. Compare the raw dispatch arguments first,
        # avoiding key normalization and hashing entirely on that common path.
        dispatch_args = self._dispatch_arg_getter(args)
        if dispatch_args == self._last_dispatch_args:
            compiled_func = self._last_compiled_func
            keys = None
        else:
            keys = extract_keys(args, self.const_params, self.tuning_params)
            compiled_func = self.dispatch_table.get(keys)

        if compiled_func is None:
            # slow path
            if keys is None:
                keys = extract_keys(args, self.const_params, self.tuning_params)
            jit_key, tuning_key = keys
            jit_instance: Optional[JitInstance] = self.jit_instances.get(jit_key, None)
            if jit_instance is None:
                jit_instance = JitInstance(self.script_cls, self.params, self.build_options, self.schedules, jit_key)
                self.jit_instances[jit_key] = jit_instance

            compiled_program = jit_instance._pick_best_program(args)
            compiled_func = compiled_program.get_launch_func()
            self.dispatch_table[(jit_key, tuning_key)] = compiled_func

        self._last_dispatch_args = dispatch_args
        self._last_compiled_func = compiled_func

        # call the compiled function
        kernel_args = self._kernel_arg_getter(args)
        ret = compiled_func(*kernel_args)

        return ret

    def compile(self, *args: Any, **kwargs: Any) -> JitInstance:
        """Compile the script for the given arguments without executing it.

        This transpiles every schedule in the autotune space into a Program and builds each Program
        to a shared library, but does not run the kernel and does not benchmark/persist a dispatch
        choice. Useful in CI to validate that a kernel compiles for a target architecture (e.g.,
        sm100a) on a machine that does not support running it. Combine with
        :func:`tilus.target.scope` to override the build target.

        Parameters
        ----------
        args:
            The positional arguments to ``__call__``.

        kwargs:
            The keyword arguments to ``__call__``.

        Returns
        -------
        jit_instance: JitInstance
            The JIT instance for the script with the given arguments. The compiled programs are
            available as ``jit_instance.valid_programs`` and ``jit_instance.compiled_programs``.
        """
        jit_instance = self._jit_instance_for(*args, **kwargs)
        jit_instance.programs()
        return jit_instance

    def _jit_instance_for(self, *args: Any, **kwargs: Any) -> JitInstance:
        if kwargs or self.with_default:
            # we allow the user to pass the keyword arguments to the script instance, or use the default values
            bound_args = self.params.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            args = bound_args.args
        else:
            if len(args) != len(self.params.param_names):
                raise ValueError(
                    "The number of arguments should be {}, but got {}.".format(len(self.params.param_names), len(args))
                )

        jit_key, _ = extract_keys(args, self.const_params, self.tuning_params)
        jit_instance: Optional[JitInstance] = self.jit_instances.get(jit_key, None)
        if jit_instance is None:
            jit_instance = JitInstance(self.script_cls, self.params, self.build_options, self.schedules, jit_key)
            self.jit_instances[jit_key] = jit_instance
        return jit_instance


class InstantiatedScriptCache:
    cache: dict[Any, InstantiatedScript] = {}

    @classmethod
    def _is_hashable(cls, obj):
        """Check if the obj is hashable."""
        try:
            hash(obj)
            return True
        except TypeError:
            return False

    @classmethod
    def _normalize_key(cls, obj):
        """Convert the obj to a hashable key."""
        if isinstance(obj, (str, int, float, bytes)):
            return obj
        elif inspect.isclass(obj):
            return obj
        elif isinstance(obj, Sequence):
            return tuple(cls._normalize_key(item) for item in obj)
        elif isinstance(obj, Mapping):
            items = []
            for key, value in obj.items():
                items.append((cls._normalize_key(key), cls._normalize_key(value)))
            items = sorted(items, key=lambda x: x[0])
            return tuple(items)
        elif cls._is_hashable(obj):
            return obj
        else:
            raise NotImplementedError(type(obj))

    @classmethod
    def get(
        cls, script_cls: Type[Script], script_args: Sequence[Any], script_kwargs: Mapping[str, Any]
    ) -> InstantiatedScript:
        key = cls._normalize_key((script_cls, script_args, script_kwargs))
        if key not in cls.cache:
            instantiated_script = InstantiatedScript(script_cls, script_args, script_kwargs)
            cls.cache[key] = instantiated_script
        return cls.cache[key]
