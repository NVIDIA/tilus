import tilus
from tilus import uint8, uint32
from tilus.extensions.hidet.ir.primitives.cuda.cvta import cvta_generic_to_shared, cvta_generic_to_cluster_shared
from tilus.extensions.hidet.ir.primitives.cuda.mapa import mapa_shared
from tilus.extensions.hidet.utils.ncu_utils import ncu_run


tilus.option.cache_dir('./cache')
tilus.option.debug.dump_ir()
tilus.utils.clear_cache()


class SharedMemoryAddressingExample(tilus.Script):
    def __call__(self):
        self.attrs.blocks = (4, 2)
        self.attrs.cluster_blocks = (4, 1)
        self.attrs.warps = 1


        a = self.shared_tensor(dtype=uint8, shape=[1024])
        a_ptr: ~uint8 = ~a[0]
        a_addr_cta: uint32 = cvta_generic_to_shared(a_ptr)
        a_addr_cluster: uint32 = cvta_generic_to_cluster_shared(a_ptr)
        a_addr_cluster_mapa: uint32 = mapa_shared(a_addr_cta, cta_rank=1 << 1)

        self.printf("[%d, %d, %d][%d] Address of a[0]: %p\n", self.blockIdx.x, self.blockIdx.y, self.blockIdx.z, self.block_rank_in_cluster, a_ptr)
        self.printf("[%d, %d, %d][%d] Address of a[0] (CTA): 0x%x\n", self.blockIdx.x, self.blockIdx.y, self.blockIdx.z, self.block_rank_in_cluster, a_addr_cta)
        self.printf("[%d, %d, %d][%d] Address of a[0] (Cluster): 0x%x\n", self.blockIdx.x, self.blockIdx.y, self.blockIdx.z, self.block_rank_in_cluster, a_addr_cluster)
        self.printf("[%d, %d, %d][%d] Address of a[0] (Cluster MAPA at cta_rank=0): 0x%x\n", self.blockIdx.x, self.blockIdx.y, self.blockIdx.z, self.block_rank_in_cluster, a_addr_cluster_mapa)


def main():
    kernel = SharedMemoryAddressingExample()
    kernel()


if __name__ == "__main__":
    main()
    # ncu_run(main)
