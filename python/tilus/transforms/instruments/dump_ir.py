from pathlib import Path
from tilus.ir.prog import Program
from tilus.ir.tools import IRPrinter
from tilus.transforms.instruments.instrument import PassInstrument


class DumpIRInstrument(PassInstrument):
    def __init__(self, dump_dir: Path):
        self.dump_dir: Path = dump_dir
        self.count: int = 0

        self.dump_dir.mkdir(parents=True, exist_ok=True)

    def before_all_passes(self, program: Program):
        printer = IRPrinter()
        with open(self.dump_dir / "0_original.txt", "w") as f:
            f.write(str(printer(program)))

        self.count = 1

    def after_pass(self, pass_name: str, program: Program):
        printer = IRPrinter()
        with open(self.dump_dir / f"{self.count}_{pass_name}.txt", "w") as f:
            f.write(str(printer(program)))

        self.count += 1
