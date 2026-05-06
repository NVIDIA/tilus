# Blackwell NVFP4 × NVFP4 → BF16 Matmul

Block-scaled FP4 × FP4 GEMM kernels for B200 (SM100), targeting the **NVFP4**
format: 4-bit operands (E2M1) with 8-bit scale factors (UE4M3, exposed in
PyTorch as `float8_e4m3fn`) covering 16-element blocks along K.

These kernels are a stepping stone toward the FP8 × FP4 mixed-precision GEMMs
in [`../blackwell_matmul_fp8xfp4/`](../blackwell_matmul_fp8xfp4/) which the
mega-MoE port needs. NVFP4 is the right place to start because:

1. **PyTorch + cuBLASLt expose NVFP4** via `torch._scaled_mm`, giving us a
   correctness *and* performance baseline (the only block-scaled FP4 mode the
   library actually wraps today).
2. The full block-scaled MMA pipeline (FP4 dtypes in HBM/SMEM, UE4M3 SFs in
   HBM/SMEM/TMEM, UTCCP SF transpose-load, block-scaled `tcgen05.mma`) is
   exercised end-to-end — once this works, the FP8 × FP4 variant only changes
   the operand dtype + a few descriptor bits.

| File | Purpose |
|------|---------|
| `matmul_v0.py` | **Tutorial / minimal end-to-end** — TMA loads + mbarrier sync + 1 warpgroup. Mirrors the style of `../blackwell_matmul/matmul_v1.py` (FP16 tutorial). |
| `matmul_v1.py` | **Fully optimized** — multi-stage pipeline + warp specialization + persistent grid + 2-CTA cluster. Mirrors the style of `../blackwell_matmul/matmul_v6.py`. *(planned)* |

Read the [Blackwell matmul tutorial](../blackwell_matmul/) (FP16 series, v0→v6) first.

## Format reference

```
A         [M, K]      element dtype = E2M1 (FP4),     packed 2-per-byte in HBM/SMEM
B         [N, K]      element dtype = E2M1 (FP4),     packed 2-per-byte in HBM/SMEM
SFA       [M, K/16]   element dtype = UE4M3 (≡ E4M3), one byte per scale
SFB       [N, K/16]   element dtype = UE4M3 (≡ E4M3), one byte per scale
C         [M, N]      element dtype = BF16,           output of D = A @ B^T after dequant
```

Block size **16 along K** is part of the NVFP4 spec — every 16 consecutive
K-elements in one M-row of A share one E4M3 scale; same for B's N-cols.

## PTX op

```ptx
tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
    [tmem_d], smem_desc_a, smem_desc_b, instr_desc, [tmem_sfa], [tmem_sfb], p;
```

i.e. `kind::mxf4nvf4` + `.scale_vec::4X` (alias `.block16`). Operands and SFs
both come from descriptors; SFs sit in TMEM and are loaded there from SMEM via
`tcgen05.cp` (UTCCP).

## Reference: PyTorch `_scaled_mm`

```python
out = torch._scaled_mm(
    a, b.t(),                                 # FP4, packed (float4_e2m1fn_x2)
    scale_a=sfa, scale_b=sfb.t(),             # UE4M3 (float8_e4m3fn)
    out_dtype=torch.bfloat16,
)
```

Constraints (from `torch._scaled_mm`'s own error message): operands
`float4_e2m1fn_x2`, scales `float8_e4m3fn`, `M × K/16` SF elements per
operand, both contiguous after transpose.

## Running

```bash
cd /home/scratch.yaoyaod_gpu/repos/tilus
python examples/blackwell_matmul_nvfp4/matmul_v0.py
```
