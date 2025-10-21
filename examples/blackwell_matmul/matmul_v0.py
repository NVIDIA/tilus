import os
import torch
import tilus
from tilus import float16, float32, int32, uint32
from tilus.utils import cdiv

tilus.option.cache_dir(os.path.join(os.path.dirname(__file__), "cache"))
tilus.option.debug.dump_ir()

tilus.target.set_current_target(tilus.target.nvgpu_sm100a)


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
        g_b = self.global_view(b_ptr, dtype=float16, shape=[n_size, k_size])
        s_a = self.shared_tensor(dtype=float16, shape=[self.block_m, self.block_k])
        s_b = self.shared_tensor(dtype=float16, shape=[self.block_n, self.block_k])
        t_acc = self.tcgen05.alloc(dtype=float32, shape=[self.block_m, self.block_n])

        mbarriers = self.mbarrier.alloc(counts=[1])
        phase: uint32 = 0

        self.print_tensor("t_acc: ", t_acc)

        for offset_k in range(0, k_size, self.block_k):
            self.copy_async(src=g_a, dst=s_a, offsets=[offset_m, offset_k])
            self.copy_async(src=g_b, dst=s_b, offsets=[offset_n, offset_k])
            self.copy_async_wait_all()
            self.sync()

            self.tcgen05.mma(s_a, s_b.transpose(), t_acc)
            self.tcgen05.commit(mbarrier=mbarriers[0])
            self.mbarrier.wait(mbarriers[0], phase=phase)
            phase ^= 1

        r_acc = self.tcgen05.load(t_acc, offsets=[0, 0], shape=[self.block_m, self.block_n])
        g_c = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
        self.store_global(g_c, r_acc.to(float16), offsets=[offset_m, offset_n])

        self.tcgen05.dealloc(t_acc)
        self.sync()


def main():
    matmul = BlackwellMatmul()

    m_size, n_size, k_size = 128, 64, 32

    a = torch.ones(m_size, k_size, dtype=torch.float16, device="cuda")
    b = torch.ones(n_size, k_size, dtype=torch.float16, device="cuda")
    c = torch.empty(m_size, n_size, dtype=torch.float16, device="cuda")

    matmul(m_size, n_size, k_size, a, b, c)
    torch.cuda.synchronize()

    c_ref = a @ b.T

    print(a)
    print(b)
    print(c_ref)
    print(c)


    torch.testing.assert_close(c, c_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    main()
