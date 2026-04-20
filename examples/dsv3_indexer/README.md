# DeepSeek V3.2 Sparse-Attention Indexer

A Tilus implementation of the **indexer** used by DeepSeek V3.2 Sparse
Attention (DSA). The indexer picks the **top-2048 most relevant key tokens**
per query; downstream attention then runs only on those 2048 tokens. Full
attention over a long context is O(L²), which is the whole reason DSA exists
— selecting 2048 out of up to ~500K keys per step is what makes decode with
128K+ context practical.

## Files in this folder

| File | What it is |
| ---- | ---------- |
| [`baseline.py`](baseline.py) | PyTorch reference implementation — correctness oracle. |
| [`benchmark.py`](benchmark.py) | Latency benchmark: PyTorch baseline / vLLM unfused pipeline / Tilus (TODO). |
| [`setup_deep_gemm.sh`](setup_deep_gemm.sh) | Clones and installs DeepGEMM into the current Python env so the `vllm` column of the benchmark can run. |
| [`requirements.txt`](requirements.txt) | `torch`, `pandas`, `vllm`. DeepGEMM is installed via the script above (not on PyPI). |
| [`README.md`](README.md) | This document. |

The reference CUDA implementation of the *fused* indexer lives in FlashInfer
PR [#2814](https://github.com/flashinfer-ai/flashinfer/pull/2814) (unmerged
at the time of writing). vLLM's production path still uses a two-kernel
*unfused* pipeline.

## What the kernel computes

Inputs (shape in brackets):

- `q` `[batch, 64, 128]` — FP8 query. 64 query heads per batch item.
- `k_cache` `[num_pages, 64, 1, 132]` — uint8 paged FP8 KV cache (layout below).
- `weights` `[batch, 64]` — float32 per-head mixing weight.
- `seq_lens` `[batch]` — int32 sequence length per row.
- `block_table` `[batch, max_num_pages]` — int32 paged-attention page indices.

For each batch item `b` with sequence length `L = seq_lens[b]`:

1. **Gather + dequantize K** from the paged FP8 cache → `K ∈ R^{L × 128}`
   (the cache has a single KV head — this is MQA, 64 Q heads share 1 K).
2. **Per-head score** `S = relu(q[b] @ K.T)` with shape `[64, L]`.
3. **Weighted sum over heads** `logits = (S * weights[b][:, None]).sum(dim=0)` → `[L]`.
4. **Top-K** — pick 2048 indices with the highest `logits`, padded with `-1`
   for rows where `L < 2048`.

Outputs:

- `indices` `[batch, 2048]` int32 — the top-K *positions* along the sequence.
  Padded with `-1` where `L < 2048`.
- `logits` `[batch, max_model_len]` float32 — the per-token score for every
  valid position `t ∈ [0, L)`. `max_model_len` is a user-supplied upper bound
  (default 163840); entries at `t ≥ L` are junk padding kept around so the
  row stride is uniform (static shape → CUDA-graph friendly, no re-alloc when
  `seq_lens` changes). Only `logits[b, :seq_lens[b]]` is meaningful.

Downstream sparse attention uses `indices` to gather K and V and runs regular
attention on those 2048 tokens. `indices` is the load-bearing output; `logits`
is returned mostly for debugging and reuse.

## KV cache layout (`[num_pages, 64, 1, 132]` uint8)

One page holds 64 tokens, each with a 128-wide head_dim in FP8 plus an FP32
scale. The declared shape `[num_pages, 64, 1, 132]` is really just a byte
count per page (`64 * 132 = 8448`). Within a page the bytes are **not**
interleaved per token — FP8 values for all 64 tokens come first, then all 64
scales:

```text
Per page (8448 bytes):
┌────────────────────────────────────────┬──────────────────────┐
│ 64 tokens × 128 FP8 bytes = 8192 bytes │ 64 × FP32 = 256 bytes│
│  → viewed as [64, 128] fp8 K values    │  → one scale / token │
└────────────────────────────────────────┴──────────────────────┘
```

The scale is broadcast over the 128-wide head_dim (one scalar per
`(page, token)` covers the whole row). See `dequant_fp8_kv_cache()` in
[`baseline.py`](baseline.py) for the exact reshape sequence.

## Why the kernel is interesting

- **MQA shape.** 64 query heads share 1 KV head, so the inner GEMM is a thin
  `64 × 128` matmul against a long K of shape `[L, 128]`. The intermediate
  `[64, L]` is potentially gigabytes for long L — a naive (non-fused)
  implementation materializes it; a fused kernel streams K through shared /
  tensor memory and accumulates the weighted sum on the fly.
- **FP8 paged cache with per-token FP32 scale** — compute in FP8, dequant to
  float before the MMA reduction.
- **Top-K over very long vectors** — up to 2¹⁹ logits per batch item; a
  radix-based histogram top-K beats `torch.topk`'s sort by a large factor.
- **Load balancing.** Low batch + long sequence has tons of per-row
  parallelism but few rows; high batch + short sequence is the opposite. The
  reference implementation precomputes an `sm_map` that partitions work
  across physical SMs.

## Top-K algorithm walkthrough

Selecting 2048 items out of up to half a million logits is the single largest
chunk of runtime in the standalone indexer, so it deserves more than a call
to `torch.topk`. The reference implementation uses **radix partition-based
selection**, a GPU-friendly variant of quickselect that runs in essentially
one pass over global memory. This section builds up the idea from scratch —
no prior top-K experience assumed.

### Why not sort?

A full sort is `O(L log L)` and throws away the fact that we only need the
top few entries. A heap-based top-K is `O(L log K)` but has an awkward inner
data structure for GPUs. Ideally we want `O(L)` with a small constant and an
inner loop that stays in shared memory. The way there is a
**partition-based** approach: split candidates into "definitely in the top-K",
"definitely out", and "contested", then recurse on just the contested bucket.
Quickselect does this with a random pivot; here we do it deterministically
using the bits of the float itself.

### Step 1: make floats comparable as unsigned integers

IEEE 754 bit order doesn't match float order — the sign bit is on top,
negatives have inverted-magnitude bits, and `-0.0 < +0.0` is a special case.
The textbook fix: flip the sign bit for positives, bitwise-NOT negatives.

```c
uint32_t bits = __float_as_uint(x);
bits = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
```

After this mapping, `unsigned_compare(a', b') == float_compare(a, b)` — larger
floats become larger uints — so top-K becomes bit-twiddling. (See
`convert_to_uint32_v2` in `include/flashinfer/dsv3_ops/common.cuh` of PR #2814.)

### Step 2: partition one byte (radix digit) at a time

A uint32 has 4 bytes. We process them from MSB to LSB in 4 passes. In pass
`t` we look at bits `[24 - 8*t, 32 - 8*t)` of each surviving candidate and:

1. **Build a 256-bin histogram** of that byte across all candidates.
2. **Find the threshold bin τ** via cumulative sum from bin 255 downward:

   ```text
   sum(hist[τ+1 .. 256)) < K_remaining ≤ sum(hist[τ .. 256))
   ```

3. **Partition** each candidate by its current byte:

   ```text
   byte > τ  → commit to final top-K         (wins on this byte; later bytes don't matter)
   byte == τ → keep for the next pass         (tie on this byte; next byte decides)
   byte < τ  → drop                           (loses on this byte)
   ```

4. **Shrink** `K_remaining -= count(byte > τ)`. The next pass's candidate
   pool is just the contested `byte == τ` bucket.

After pass 3 anything still in the pool has equal bits across all 4 bytes —
i.e. equal floats — and is committed until `K_remaining == 0`.

### Tiny worked example

Take `L = 12` values and `K = 4`. Imagine these are already uint-mapped;
we're looking at one byte:

```text
[220, 220, 215, 215, 210, 210, 210, 200, 195, 180, 150, 50]
```

Histogram (one byte): `220:2  215:2  210:3  200:1  195:1  180:1  150:1  50:1`.

Cumulative from the top: `sum[≥ 221] = 0`, `sum[≥ 220] = 2`, `sum[≥ 216] = 2`,
`sum[≥ 215] = 4`. We need `K = 4`. The smallest bin where `cum ≥ K` is `215`,
and the bin just above (`216`) has `cum = 2 < 4`, so **τ = 215**.

Partition:

- `byte > 215` → commit `{220, 220}` (2 elements → `K_remaining = 2`).
- `byte == 215` → keep `{215, 215}` for next pass.
- `byte < 215`  → drop.

Next pass runs on just the 2 kept entries with the next byte. If they tie
again we eventually fall through to the "equal floats" mop-up. Final
top-4 = `{220, 220, 215, 215}`. The real kernel does this 4 times (once per
byte of a 32-bit key).

### Why this is so fast

**Only pass 0 touches the full `L` in global memory.** After pass 0 the
contested pool is roughly `L / 256` in expectation — small enough to fit in
a shared-memory cache of 4096 entries (the default `num_cached`). Passes
1–3 walk only that cache → zero global-memory traffic after pass 0. The
whole algorithm is essentially *one sweep over L* plus tiny shared-memory
shuffles.

If pass 0's histogram is supplied by the upstream logits kernel (the
`PRE_HISTOGRAM=true` path), it saves one more global scan: the MQA GEMM's
epilogue already atomic-adds into the byte-3 histogram as a side effect, so
the fused top-K skips "scan L to build histogram" and goes straight to
"scan L to partition". This is the **histogram fusion** that FlashInfer
PR [#2814](https://github.com/flashinfer-ai/flashinfer/pull/2814) introduces
on top of a standalone radix top-K.

### Multi-CTA clusters

For short `L` a single CTA can't fill the GPU. The kernel launches
`NClusters ∈ {2, 4, 8, 16}` CTAs per batch row as a **thread-block cluster**
(SM90+ feature). Each CTA strides over a disjoint slice of `logits` with step
`NClusters × 1024`. Histograms are summed cluster-wide through **distributed
shared memory**:

```cpp
// CTA A reads peer CTA's shared histogram directly, no global round-trip
cum_val += cluster.map_shared_rank(&shared_hist[...][threadIdx.x], peer_rank)[0];
```

Each CTA keeps its own candidate cache and its own local top-K list; at the
end, CTAs `block_id > 0` atomic-add into CTA 0's counter to get a starting
offset and write their slice into the final output. The Python-side API
picks cluster size by batch: bigger batch → fewer clusters per row, since
inter-row parallelism already saturates.

### Approximate vs. exact (overflow handling)

What if the contested bucket is larger than `num_cached` (4096)? Two
variants exist in the reference CUDA:

- **Fast** (`fast_topk_clusters.cu` in PR #2814) — silently drops overflow.
  The kept entries are the ones the grid-strided scan visits first, so the
  fast variant is **biased toward earlier sequence positions**. Usually OK
  for autoregressive decoding, but not strictly correct.
- **Exact** (`fast_topk_clusters_exact.cu` in PR #2814) — spills overflow
  to a per-CTA global-memory scratch buffer
  (`cached_overflow`, size `overflow_stride` per phase, ping-ponged across
  passes). Subsequent passes process both the shared cache *and* the global
  spill. Critically, the histogram for the next byte is still updated in
  shared memory for spilled entries, so the next pass's threshold bin stays
  correct. Measured overhead 1–6% on synthetic uniform data — real peaky
  logits may push this higher.

### Where to go next in the reference CUDA

All paths below are relative to the FlashInfer tree on PR
[#2814](https://github.com/flashinfer-ai/flashinfer/pull/2814):

- `csrc/dsv3_ops/fast_topk_clusters.cu` — the fast variant, ~350 lines; the
  core algorithm.
- `csrc/dsv3_ops/fast_topk_clusters_exact.cu` — the exact variant; diff vs.
  fast is the overflow-spill path.
- `csrc/dsv3_ops/indexer_paged_logits.cu` — the fused MQA-logits kernel
  (Blackwell SM100 / tcgen05).
- `include/flashinfer/dsv3_ops/common.cuh` — `convert_to_uint32_v2`,
  `cum_sum`, `run_vectorized`.

### Further reading

- **Dr.Top-k** (Gao et al., SC'21) — delegate-centric radix top-K on GPU.
- **RadiK** (Li et al., ICS'24) — radix-select with a cached candidate pool,
  very close in spirit to this implementation.
- **CUB `DeviceRadixSort`** docs — same monotone float-to-uint trick.

## Running the benchmark

```bash
cd examples/dsv3_indexer
python benchmark.py

# specific configs
python benchmark.py --batch-sizes 1 4 32 --seq-lens 4096 32768
```

The benchmark emits up to three columns:

| Column | What it measures | Requires |
| ------ | ---------------- | -------- |
| `baseline` | PyTorch reference (`dsv3_topk_indexer_ref` with `torch.topk`) — correctness oracle, very slow. | `torch` |
| `vllm` | **The primary comparison.** Mirrors vLLM's DSV3.2 decode indexer pipeline (see [`vllm/model_executor/layers/sparse_attn_indexer.py`](../../../.venv/lib/python3.12/site-packages/vllm/model_executor/layers/sparse_attn_indexer.py)): `vllm.utils.deep_gemm.fp8_paged_mqa_logits` for MQA logits followed by `torch.ops._C.top_k_per_row_decode` or `torch.ops._C.large_context_topk` (vLLM's own radix top-K, picked by a batch/seq_len heuristic — *not* `flashinfer.top_k`). No histogram fusion; two separate kernels back to back. | `vllm` + DeepGEMM, SM90 or SM100/SM103 GPU |
| `tilus` | Tilus implementation. **TODO** — will be added once written. | — |

Each column auto-skips if its dependency is missing. `python benchmark.py`
always runs `baseline`; the `vllm` column needs vLLM, DeepGEMM, and a
supported GPU.

### Installing DeepGEMM

DeepGEMM is not on PyPI. Use the helper script:

```bash
source /path/to/your/venv/bin/activate
./setup_deep_gemm.sh
```

The script clones [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
with `--recursive` into `third_party/deep_gemm/` (gitignored), builds the
wheel with `python setup.py bdist_wheel`, and pip-installs it into the
active interpreter. Pass `PYTHON=/path/to/python` to override interpreter
selection. The script bootstraps pip with `ensurepip` for uv-created venvs
that skip pip by default.

### GPU compatibility

DeepGEMM's `fp8_paged_mqa_logits` only supports Hopper (SM90) and datacenter
Blackwell (SM100 / SM103). Workstation / consumer Blackwell (SM120 / SM121,
e.g. RTX PRO 6000 / RTX 5090) uses a different tensor-core ISA (no tcgen05)
and is not supported by DeepGEMM today. On such GPUs the benchmark prints a
notice and only emits the `baseline` column — the Tilus implementation is
the only path forward there.

## TODO

- **Tilus implementation.** The main deliverable. Target: reproduce at least
  the fused logits kernel plus a compatible top-K in Tilus. Start with a
  scaffold in this folder (e.g. `tilus_kernel.py`) and wire it into
  `benchmark.py` under the `tilus` column.
- **Fused-kernel comparison column.** Once FlashInfer PR
  [#2814](https://github.com/flashinfer-ai/flashinfer/pull/2814) merges and
  vLLM adopts it as a replacement for the two-kernel flow, add a
  `flashinfer_fused` column so the Tilus kernel can be compared against the
  fused CUDA upper bound.

## References

- DeepSeek V3.2 (DSA) — <https://github.com/deepseek-ai/DeepSeek-V3>
- FlashInfer PR #2814 (fused indexer kernel) — <https://github.com/flashinfer-ai/flashinfer/pull/2814>
- DeepGEMM — <https://github.com/deepseek-ai/DeepGEMM>
- vLLM DSV3.2 indexer code — [`vllm/model_executor/layers/sparse_attn_indexer.py`](../../../.venv/lib/python3.12/site-packages/vllm/model_executor/layers/sparse_attn_indexer.py)
