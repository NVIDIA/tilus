from hidet.ir.dtypes import int32
from hidet.ir.expr import Var
from hidet.ir.primitives.cuda.vars import block_idx, block_dim, thread_idx, grid_dim
from hidet.ir.primitives.vars import register_primitive_variable, lookup_primitive_variable
from hidet.utils import initialize

class Dim3:
    def __init__(self, x: Var, y: Var, z: Var):
        self.x: Var = x
        self.y: Var = y
        self.z: Var = z
    
    def __repr__(self):
        return f'Dim3(x={self.x}, y={self.y}, z={self.z})'

    def __iter__(self):
        return iter((self.x, self.y, self.z))


@initialize()
def register_cuda_primitive_variables():
    for base in ['clusterBlockIdx', 'clusterBlockRank', 'clusterDim', 'clusterSize', 'clusterIdx']:
        if base == 'clusterBlockRank' or base == 'clusterSize':
            register_primitive_variable(name=base, dtype=int32)
        else:
            for suffix in ['x', 'y', 'z']:
                name = '{}_{}'.format(base, suffix)
                register_primitive_variable(name=name, dtype=int32)

def cluster_block_idx(dim: str) -> Var:
    return lookup_primitive_variable('clusterBlockIdx_{}'.format(dim))

def cluster_block_rank() -> Var:
    return lookup_primitive_variable('clusterBlockRank')

def cluster_idx(dim: str) -> Var:
    return lookup_primitive_variable('clusterIdx_{}'.format(dim))

def cluster_dim(dim: str) -> Var:
    return lookup_primitive_variable('clusterDim_{}'.format(dim))

def cluster_size() -> Var:
    return lookup_primitive_variable('clusterSize')

blockIdx = Dim3(block_idx('x'), block_idx('y'), block_idx('z'))
blockDim = Dim3(block_dim('x'), block_dim('y'), block_dim('z'))
threadIdx = Dim3(thread_idx('x'), thread_idx('y'), thread_idx('z'))
gridDim = Dim3(grid_dim('x'), grid_dim('y'), grid_dim('z'))
clusterBlockIdx = Dim3(cluster_block_idx('x'), cluster_block_idx('y'), cluster_block_idx('z'))
clusterBlockRank = cluster_block_rank()
clusterDim = Dim3(cluster_dim('x'), cluster_dim('y'), cluster_dim('z'))
clusterIdx = Dim3(cluster_idx('x'), cluster_idx('y'), cluster_idx('z'))
clusterSize = cluster_size()
