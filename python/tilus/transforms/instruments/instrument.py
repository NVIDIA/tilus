from tilus.ir.prog import Program


class PassInstrument:
    def before_all_passes(self, program: Program):
        pass

    def before_pass(self, pass_name: str, program: Program):
        pass

    def after_pass(self, pass_name: str, program: Program):
        pass

    def after_all_passes(self, program: Program):
        pass
