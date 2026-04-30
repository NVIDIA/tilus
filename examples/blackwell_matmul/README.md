# Blackwell Matmul Examples

Progressive matrix multiplication kernels for NVIDIA Blackwell GPUs, from minimal to vendor-library-level performance.

| File | Optimization |
|------|-------------|
| `matmul_v0.py` | Minimal kernel: tcgen05 MMA, mbarrier, tensor memory |
| `matmul_v1.py` | TMA loads and TMA epilogue |
| `matmul_v2.py` | Multi-stage software pipelining |
| `matmul_v3.py` | Warp specialization (TMA + MMA warps) |
| `matmul_v4.py` | Tile rasterization, Pipeline abstraction |
| `matmul_v5.py` | CLC persistent kernel, pipelined epilogue |
| `matmul_v6.py` | 2-CTA cluster, distributed MMA |

See the [tutorial](https://nvidia.github.io/tilus/tutorials/matmul-blackwell/) for detailed explanations.

## Benchmark

```bash
cd examples/blackwell_matmul

# Latency-based benchmark (all versions)
python benchmark.py

# NCU profiling (all versions)
python benchmark.py --ncu

# Specific versions and workload size
python benchmark.py --versions v2 v3 v4 --size 10240 10240 10240
```
