from hidet.ir.expr import Var

class Dim3:
    def __init__(self, x: Var, y: Var, z: Var):
        self.x: Var = x
        self.y: Var = y
        self.z: Var = z
    
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def to_tuple(self):
        return (self.x, self.y, self.z)
