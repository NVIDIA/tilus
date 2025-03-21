from pathlib import Path

from hidet.runtime.compiled_module import CompiledModule, compiled_module_exists


class CompiledProgram:
    def __init__(self, program_dir: str | Path):
        self.program_dir: Path = Path(program_dir)
        self.compiled_module = CompiledModule(str(self.program_dir / "module"))

    def __call__(self, *args):
        return self.compiled_module(*args)


def load_compiled_program(program_dir: str | Path) -> CompiledProgram:
    """
    Load a compiled program from the cache directory.

    Parameters
    ----------
    program_dir: str or Path
        The cache directory of the compiled program.

    Returns
    -------
    compiled_program: CompiledProgram
        The compiled program.
    """
    return CompiledProgram(program_dir)


def compiled_program_exists(cache_dir: Path | str) -> bool:
    """
    Check if there is a program that has been built and cached under the given program cache dir.

    Parameters
    ----------
    cache_dir: Path | str
        The cache directory of the compiled program.

    Returns
    -------
    ret: bool
        True if the program exists, False otherwise.
    """
    path = Path(cache_dir)
    return all(
        [compiled_module_exists(str(path / "module")), (path / "program.txt").exists(), (path / "options.txt").exists()]
    )
