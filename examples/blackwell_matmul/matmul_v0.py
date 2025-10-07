import tilus
from tilus import float16, float32, int32


class BlackwellMatmul(tilus.Script):
    def __init__(self):
        super().__init__()
        self.block_m = 128
        self.block_n = 64
        self.block_k = 16
    
    def __call__(
        self, 
        m_size: int32,
        n_size: int,
        k_size: int,
        a_ptr: ~float16,
        b_ptr: ~float16,
        c_ptr: ~float16,
    ):
        self.attrs.blocks = [cdiv(m_size, self.block_m), cdiv(n_size, self.block_n)]
        self.attrs.warps = 4

        offset_m: int32 = self.block_m * self.blockIdx.x
        offset_n: int32 = self.block_n * self.blockIdx.y

        g_a = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
        g_b = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])
        s_a = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_k])
        s_b = self.shared_tensor(dtype=float16, shape=[self.block_n, self.block_k])
        acc = self.tcgen05.alloc(dtype=float32, shape=[self.block_m, self.block_n])

        mbarriers = self.mbarrier.alloc(counts=[1])
        phase: uint32 = 0

        for offset_k in range(0, k_size, self.block_k):
            self.copy_async(src=g_a, dst=s_a, offsets=[offset_m, offset_k])
            self.copy_async(src=g_b, dst=s_b, offsets=[offset_n, offset_k])
            self.copy_async_wait_all()
            self.sync()

            self.tcgen05.mma(s_a, s_b.transpose(), acc)
            self.tcgen05.commit(mbarrier=mbarriers[0])
            self.mbarrier.wait(mbarriers[0], phase=phase)
            phase ^= 1
        
        self.tcgen05.dealloc(acc)
        self.mbarrier.wait(mbarriers[0], phase=phase)
        self.sync()


def main():
    matmul = BlackwellMatmul()


if __name__ == "__main__":
    main()
