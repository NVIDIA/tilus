from typing import Optional, List
import os
import shutil
from tilus.ir.func import Function
from .base import Pass
from .bound_aware_simplify import bound_aware_simplify_pass


def optimize_vm_program(
    program: Function,
    *,
    transforms: Optional[List[Pass]] = None,
    dump_dir: Optional[str] = None,
) -> Function:
    if transforms is None:
        transforms = [bound_aware_simplify_pass()]

    for idx, transform in enumerate(transforms):
        if dump_dir and idx == 0:
            shutil.rmtree(dump_dir, ignore_errors=True)
            os.makedirs(dump_dir, exist_ok=True)
            with open(os.path.join(dump_dir, "0_Original.txt"), "w") as f:
                f.write(str(program))

        program = transform(program)

        if dump_dir:
            transform_name = transform.__class__.__name__.removesuffix("Pass")
            with open(os.path.join(dump_dir, f"{idx + 1}_{transform_name}.txt"), "w") as f:
                f.write(str(program))

    return program
