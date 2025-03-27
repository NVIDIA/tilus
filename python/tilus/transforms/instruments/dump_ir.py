import shutil
import time
from pathlib import Path

import tabulate

from tilus.ir.prog import Program
from tilus.ir.tools import IRPrinter
from tilus.transforms.instruments.instrument import PassInstrument


class DumpIRInstrument(PassInstrument):
    def __init__(self, dump_dir: Path):
        self.dump_dir: Path = dump_dir
        self.count: int = 0
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.lower_time_path: Path = self.dump_dir / "lower_time.txt"
        self.start_time: dict[str, float] = {}
        self.elapsed_time: dict[str, float] = {}

    def before_all_passes(self, program: Program) -> None:
        printer = IRPrinter()
        # remove the old dump directory
        shutil.rmtree(self.dump_dir, ignore_errors=True)

        self.dump_dir.mkdir(parents=True, exist_ok=True)
        with open(self.dump_dir / "0_Original.txt", "w") as f:
            f.write(str(printer(program)))

        self.count = 1

    def before_pass(self, pass_name: str, program: Program) -> None:
        self.start_time[pass_name] = time.time()

    def after_pass(self, pass_name: str, program: Program) -> None:
        self.elapsed_time[pass_name] = time.time() - self.start_time[pass_name]

        printer = IRPrinter()
        with open(self.dump_dir / f"{self.count}_{pass_name}.txt", "w") as f:
            f.write(str(printer(program)))

        self.count += 1

    def after_all_passes(self, program: Program) -> None:
        headers = ["Pass", "Time"]
        rows = []
        for name, elapsed_time in self.elapsed_time.items():
            rows.append([name, "{:.3f} seconds".format(elapsed_time)])
        with open(self.lower_time_path, "w") as f:
            f.write(tabulate.tabulate(rows, headers=headers))
