from tilus.ir.inst import MmaConfig


class cuda:
    class mma:
        m16n8k16_f16_f32 = MmaConfig.m16n8k16_f16_f32()
        m16n8k16_f16_f16 = MmaConfig.m16n8k16_f16_f16()
        m16n8k16_bf16_f32 = MmaConfig.m16n8k16_bf16_f32()
