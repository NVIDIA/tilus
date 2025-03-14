from tilus.ir.program import VirtualMachineProgram


class VirtualMachinePass:
    def __call__(self, prog: VirtualMachineProgram) -> VirtualMachineProgram:
        raise NotImplementedError()
