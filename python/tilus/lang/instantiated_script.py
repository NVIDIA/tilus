import hashlib
import inspect
import json
import os
import pickle
import traceback
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Type

import filelock
import tabulate

import tilus.option
from hidet.ir import Constant
from hidet.ir.type import DataType, PointerType
from hidet.runtime import CompiledFunction
from hidet.utils.py import nocolor
from tilus.drivers import BuildOptions, build_program, get_cache_dir
from tilus.extensions.hidet.ir.expr import as_expr
from tilus.ir.prog import Program
from tilus.lang.script import Script
from tilus.runtime import CompiledProgram, load_compiled_program
from tilus.utils import benchmark_func, relative_to_with_walk_up, to_snake_case
from tilus.utils.multiprocess import parallel_imap


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
            raise TypeError("Could not instantiate the script with the arguments and tuning arguments: " + str(e))
        schedule = dict(bound_args.arguments)
        schedule.pop("self")  # remove the 'self' argument
        schedules.append(schedule)
    return schedules


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
            if isinstance(param.annotation, (DataType, PointerType)) or param.annotation in [bool, int, float, str]:
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
    multiples = [1, 8, 16, 32]
    for n in range(max(multiples)):
        for m in reversed(multiples):
            if n % m == 0:
                divisibility_key_list.append(m)
                break
        else:
            assert False
    divisibility_key = tuple(divisibility_key_list)


_init_divisibility_key()


def extract_keys(args: Sequence[Any], const_params: list[int], tuning_params: list[int]) -> tuple[JitKey, TuningKey]:
    jit_key = []
    tuning_key = []
    for i in const_params:
        jit_key.append(args[i])
    for i in tuning_params:
        arg: int = args[i]
        jit_key.append(divisibility_key[arg % 32])
        arg_block = 2 ** max((arg.bit_length() - 2), 0)
        tuning_key.append((arg - arg_block + 1) // arg_block * arg_block)
    return tuple(jit_key), tuple(tuning_key)


class JitInstance:
    def __init__(
        self,
        script_cls: Type[Script],
        call_params: CallParameters,
        build_options: BuildOptions,
        schedules: list[dict[str, Any]],
        jit_key: JitKey,
    ):
        self.script_cls: Type[Script] = script_cls
        self.call_params: CallParameters = call_params
        self.build_options: BuildOptions = build_options
        self.schedules: list[dict[str, Any]] = schedules
        self.jit_key: JitKey = jit_key

        self.programs: list[Program] = []
        self.valid_schedules: list[int] = []
        self.compiled_programs: list[CompiledProgram] = []
        self.failed_scheduling: list[str] = []
        self.failed_building: list[tuple[str, str]] = []
        self.dispatch_table: dict[TuningKey, int] = {}
        self.cache_dir: Path = Path()

        self.initialize()

        # some checks
        if script_cls.debug_block and len(self.compiled_programs) > 1:
            raise ValueError("Please specify the debug_schedule when debug_block is set. ")

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
            name2consts: dict[str, Constant] = {}
            for idx in call_params.const_params:
                name = call_params.param_names[idx]
                const = as_expr(name2const_py[name])
                assert isinstance(const, Constant)
                name2consts[name] = const

            function = transpiler.transpile(script_obj, name2consts=name2consts, name2divisibility=name2divisibility)
            function = function.with_name(to_snake_case(script_cls.__name__))
            program = Program.create(functions={function.name: function})
            return program
        except Exception:
            return traceback.format_exc()

    @staticmethod
    def _build_program(job: Any) -> str:
        assert len(job) == 2
        program: Program = job[0]
        options: BuildOptions = job[1]
        try:
            build_program(program, options=options, load=False)
        except Exception:
            return traceback.format_exc()
        else:
            return "success"

    def initialize(self):
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
        programs: list[Program] = []
        schedules: list[int] = []
        for idx, item in enumerate(parallel_imap(func=JitInstance._instantiate_schedule, jobs=scheduling_jobs)):
            if isinstance(item, Program):
                programs.append(item)
                schedules.append(idx)
            else:
                sections = {"Schedule": str(self.schedules[idx]), "Traceback": nocolor(item)}
                lines = []
                for key, value in sections.items():
                    lines.append(key)
                    lines.append("=" * len(key))
                    lines.append("")
                    lines.append(value)
                    lines.append("")
                self.failed_scheduling.append("\n".join(lines))

        # build the programs in parallel
        building_jobs = [(program, self.build_options) for program in programs]
        for idx, item in enumerate(parallel_imap(func=JitInstance._build_program, jobs=building_jobs)):
            program_cache_dir = get_cache_dir(programs[idx], options=self.build_options)
            if item == "success":
                self.programs.append(programs[idx])
                self.valid_schedules.append(schedules[idx])
                self.compiled_programs.append(load_compiled_program(program_cache_dir))  #
            else:
                sections = {
                    "Schedule": str(self.schedules[idx]),
                    "Program": str(programs[idx]),
                    "Traceback": nocolor(item),
                }
                lines = []
                for key, value in sections.items():
                    lines.append(key)
                    lines.append("=" * len(key))
                    lines.append("")
                    lines.append(value)
                    lines.append("")
                self.failed_building.append(("\n".join(lines), str(program_cache_dir)))

        # get the cache dir for this jit instance
        concatenated_program_text = "\n".join([str(program) for program in self.programs])
        option_text = str(self.build_options)
        hash_string = option_text + concatenated_program_text
        hash_key = hashlib.sha256(hash_string.encode()).hexdigest()[:8]
        jit_name_items = [to_snake_case(self.script_cls.__name__)] + [str(key) for key in self.jit_key] + [hash_key]
        self.cache_dir = Path(tilus.option.get_option("cache_dir")) / "scripts" / "-".join(jit_name_items)

        # initialize the script cache if it does not exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_dir / ".lock"
        with filelock.FileLock(lock_path):
            # add soft link to the cache dirs of compiled programs
            programs_dir = self.cache_dir / "programs"
            if not programs_dir.exists():
                programs_dir.mkdir()
                for idx, compiled_program in enumerate(self.compiled_programs):
                    start_path = (programs_dir / str(idx)).resolve()
                    target_path = compiled_program.program_dir.resolve()
                    relative_path = relative_to_with_walk_up(start_path.parent, target=target_path)
                    os.symlink(relative_path, start_path)

            # write the schedule.txt
            schedule_path = self.cache_dir / "schedule.txt"
            if not schedule_path.exists() and self.valid_schedules:
                headers = ["index"] + list(self.schedules[self.valid_schedules[0]].keys())
                rows = []
                for idx in self.valid_schedules:
                    rows.append([idx] + list(self.schedules[idx].values()))
                with open(schedule_path, "w") as f:
                    f.write(tabulate.tabulate(rows, headers=headers))

            # write the failed schedules to the cache dir
            failed_scheduling_dir = self.cache_dir / "failed" / "scheduling"
            if not failed_scheduling_dir.exists() and self.failed_scheduling:
                failed_scheduling_dir.mkdir(parents=True, exist_ok=True)
                for idx, failed_schedule in enumerate(self.failed_scheduling):
                    with open(failed_scheduling_dir / f"{idx}.txt", "w") as f:
                        f.write(failed_schedule)
            failed_building_dir = self.cache_dir / "failed" / "building"
            if not failed_building_dir.exists() and self.failed_building:
                failed_building_dir.mkdir(parents=True, exist_ok=True)
                for idx, (failed_building, prog_cache_dir) in enumerate(self.failed_building):
                    with open(failed_building_dir / f"{idx}.txt", "w") as f:
                        f.write(failed_building)
                    start_path = (failed_building_dir / f"{idx}").resolve()
                    target_path = Path(prog_cache_dir).resolve()
                    relative_path = relative_to_with_walk_up(start_path.parent, target=target_path)
                    os.symlink(relative_path, start_path)

            # write the source code of the script class to source.txt
            source_path = self.cache_dir / "source.txt"
            if not source_path.exists():
                with open(source_path, "w") as f:
                    f.write(inspect.getsource(self.script_cls))

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

        # load the dispatch table from the cache dir
        self.load_dispatch_table()

        if len(self.compiled_programs) == 0:
            raise RuntimeError(
                "No valid schedule found. Please check the following cache dir for failure information: \n"
                "    " + str(self.cache_dir.resolve())
            )

    def pick_best_program(self, args: Sequence[Any]) -> CompiledProgram:
        _, tuning_key = extract_keys(args, self.call_params.const_params, self.call_params.tuning_params)

        # check if the tuning key exists in the dispatch table
        if tuning_key not in self.dispatch_table:
            latency: list[float] = []
            if len(self.compiled_programs) == 1:
                # we skip the benchmark if there is only one compiled program
                latency.append(0.0)
            else:
                for compiled_program in self.compiled_programs:
                    kernel_args = [args[i] for i in self.call_params.kernel_params]
                    compiled_func = compiled_program.compiled_module.functions["launch"]
                    latency.append(benchmark_func(lambda: compiled_func(*kernel_args), warmup=1, repeat=10))  # type: ignore

            best_latency = min(latency)
            self.dispatch_table[tuning_key] = latency.index(best_latency)
            self.dump_dispatch_table()

            # write the benchmark results to timing
            timing_dir = self.cache_dir / "timing"
            tuning_report_path = timing_dir / "{}.txt".format("-".join(str(k) for k in tuning_key))
            if not timing_dir.exists():
                timing_dir.mkdir()
            with open(tuning_report_path, "w") as f:
                headers = ["index"] + list(self.schedules[self.valid_schedules[0]].keys()) + ["latency (ms)"]
                rows = []
                for i in range(len(self.valid_schedules)):
                    schedule_values = list(self.schedules[self.valid_schedules[i]].values())
                    rows.append([i] + schedule_values + [latency[i]])
                rows = sorted(rows, key=lambda x: x[-1])
                with open(tuning_report_path, "w") as f:
                    f.write(tabulate.tabulate(rows, headers=headers, floatfmt=".2f"))

        return self.compiled_programs[self.dispatch_table[tuning_key]]

    def load_dispatch_table(self):
        table_path = self.cache_dir / "dispatch_table.json"
        if table_path.exists():
            with open(table_path, "r") as f:
                self.dispatch_table = json.load(f)

    def dump_dispatch_table(self):
        table_path = self.cache_dir / "dispatch_table.pickle"
        table_txt_path = self.cache_dir / "dispatch_table.txt"
        with open(table_path, "wb") as f:
            pickle.dump(self.dispatch_table, f)
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
        self.dispatch_table: dict[tuple[JitKey, TuningKey], CompiledFunction] = {}

    def __call__(self, *args, **kwargs):
        if kwargs or self.with_default:
            # we allow the user to pass the keyword arguments to the script instance, or use the default values
            bound_args = self.params.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            args = bound_args.args

        # extract the JIT key and the tuning key
        jit_key, tuning_key = extract_keys(args, self.const_params, self.tuning_params)

        # check if the compiled function exists
        compiled_func: Optional[CompiledFunction] = self.dispatch_table.get((jit_key, tuning_key), None)

        if compiled_func is None:
            # slow path
            jit_instance: Optional[JitInstance] = self.jit_instances.get(jit_key, None)
            if jit_instance is None:
                jit_instance = JitInstance(self.script_cls, self.params, self.build_options, self.schedules, jit_key)
                self.jit_instances[jit_key] = jit_instance
            compiled_program = jit_instance.pick_best_program(args)
            compiled_func = compiled_program.compiled_module.functions["launch"]
            self.dispatch_table[(jit_key, tuning_key)] = compiled_func

        # call the compiled function
        kernel_args = (args[i] for i in self.kernel_params)
        return compiled_func(*kernel_args)
