class utils:
    @staticmethod
    def ceil_div(a, b):
        return (a + b - 1) // b

    @staticmethod
    def round_up_to_multiple(a, b):
        return ((a + b - 1) // b) * b
