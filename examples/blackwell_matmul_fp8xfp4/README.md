# FP8 × FP4 matmul on Blackwell (SM100)

Working folder for the **dual-matmul / block-scaled FP8×FP4 GEMM** piece of
the mega-MoE port.

The end goal is to match
[DeepGEMM `m_grouped_fp8_fp4_gemm_nt_contiguous`](../../experiments/megamoe/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh)
performance — that is the building block mega-MoE composes into its
`L1 + SwiGLU + L2` fused pipeline.

## Roadmap

| Step | What | Status |
|---|---|---|
| 1 | Single-MMA mixed-precision (`kind::f8f6f4`, no SF) | ⏳ in this folder, see `mma_fp8xfp4.py` |
| 2 | Single-MMA block-scaled (`kind::mxf8f6f4`, UE8M0 SF per-32-K) | needs UE8M0 dtype + UTCCP + scaled MMA in tilus |
| 3 | Tile-level GEMM with persistent grid + warp specialization | reuse the matmul_v6.py skeleton, swap MMA op |
| 4 | M-grouped (per-expert contiguous M) GEMM | matches `m_grouped_fp8_fp4_gemm_nt_contiguous` |
| 5 | Composed L1 + SwiGLU + L2 within one kernel | the "dual matmul" component of mega-MoE |

## Files

- `mma_fp8xfp4.py` — Step 1 (works today, modulo any FP4 packing fixes we hit)
  + Step 2 scaffold (`Tcgen05MmaMxF8F6F4Skeleton`) marking the call sites that
  will change once block-scaled MMA support lands.

## Running

```
cd /home/scratch.yaoyaod_gpu/repos/tilus
pytest examples/blackwell_matmul_fp8xfp4/mma_fp8xfp4.py -v
```

## References

- [PTX ISA: tcgen05.mma](../../experiments/megamoe/knowledge/ptx-isa-markdown/cuda_skill/references/ptx-docs/9-instruction-set/9.7.16.10-tensorcore-5th-generation-matrix-multiply-and-accumulate-operations.md)
- [PTX ISA: matrix descriptors / atype-btype encoding](../../experiments/megamoe/knowledge/ptx-isa-markdown/cuda_skill/references/ptx-docs/9-instruction-set/9.7.16.4-matrix-descriptors.md)
- DeepGEMM ref kernels:
  [single GEMM](../../experiments/megamoe/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh),
  [PTX wrappers](../../experiments/megamoe/DeepGEMM/deep_gemm/include/deep_gemm/ptx/tcgen05.cuh)
- Tilus dense-MMA reference: `examples/blackwell_matmul/matmul_v6.py`
