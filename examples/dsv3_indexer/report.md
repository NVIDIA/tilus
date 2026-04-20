# Porting the DSV3.2 Sparse-Attention Indexer to Tilus

## 0. TL;DR

The indexer is two kernels back-to-back (plus a metadata precompute):

1. **`indexer_paged_logits`** — fused MQA logits: FP8 paged K × FP8 Q → per-token FP32 logit, with the top byte of each logit atomic-added into a per-batch 256-bin histogram as a free side effect.
2. **`fast_topk_clusters`** (or `_exact`) — radix top-2048 over up to ~500K logits per batch item, using a 4-byte, 4-pass partition. Pass 0 can reuse the histogram from the epilogue above.
3. **`indexer_metadata`** — tiny helper that computes an `sm_map` so each physical SM knows which batch item / page range it owns.

**Feasibility in Tilus today:**

| Kernel | Feasibility | Main blocker |
|---|---|---|
| Logits (MMA + weighted reduce, no histogram) | Doable, close to `blackwell_matmul/matmul_v1.py` + `attention_with_kvcache` paged pattern | None critical; manual FP8 dequant with per-token scale. |
| Logits **with histogram-fusion epilogue** | Blocked | Needs `self.atomic.{shared,global}_add` tile instructions. |
| Radix top-K (single CTA, fast variant) | Blocked | Needs the atomic family, `scatter_add`, non-atomic `store_shared_scatter`, and warp-level `shfl`/`ballot`/lane intrinsics. |
| Radix top-K, `NClusters > 1` (distributed histogram) | Blocked | Needs `self.cluster.map_shared(smem, target_rank)` + remote-shared load/store/atomic (all routed through `mapa.shared::cluster`). |
| `_exact` variant (global overflow spill) | Blocked | Needs global scatter store + global atomics. |
| `indexer_metadata` | Awkward but doable | It's a tiny 1-block kernel doing `cg::reduce` + `exclusive_scan` + `atomicAdd`. Currently Tilus has no cooperative-groups warp scan/reduce primitives. Simpler to keep it as a CUDA kernel (via hidet) or do the scan on CPU. |

In short, the logits half can land today with only moderate work; **the top-K half requires new Tilus instructions**. Concrete list in §4.

---

## 1. What tilus gives us today

### 1.1 Programming model (already well-suited)

- Scripts declare `self.attrs.blocks`, `self.attrs.cluster_blocks`, `self.attrs.warps`. All three are first-class — [blackwell_matmul/matmul_v6.py](../blackwell_matmul/matmul_v6.py) uses `cluster_blocks = 2`, demonstrating the two-CTA path.
- Three tensor spaces: `register_tensor`, `shared_tensor`, and `tcgen05.alloc` (TMEM). Global access via `global_view(ptr, dtype, shape, strides)` — exactly what the indexer needs for `Q`, `K_cache`, `weights`, `seq_lens`, `block_table`, and `logits`/`indices` outputs.
- Paged addressing is clean: [examples/attention_with_kvcache/attention_v1.py:126-140](../attention_with_kvcache/attention_v1.py#L126-L140) reads `g_block_table[bs, kv_offset // page_block_size].item()` and uses the result as a scalar leading offset to `copy_async`. **Exactly the pattern the indexer needs** for `K_cache[block_table[b, page_id], ...]`.
- `thread_group(begin, num_threads)` + `single_thread()` + `single_warp()` + `warp_group()` cover the warp-specialized patterns from the CUDA reference (LDG / LDG-scales / MMA / Epilogue warps).

### 1.2 Instructions we *do* have that matter for this kernel

| Need | Tilus has | File |
|---|---|---|
| Async global → shared bulk (TMA) | `self.tma.global_to_shared(...)` with mbarrier + tx-count + multicast | [python/tilus/lang/instructions/tma.py](../../python/tilus/lang/instructions/tma.py) |
| Classic `cp.async` (SM80) | `copy_async`, `copy_async_commit_group/wait_group` | root.py:672 |
| Blackwell tensor-core MMA | `tcgen05.alloc`, `tcgen05.copy`, `tcgen05.mma`, `tcgen05.load/store`, `tcgen05.commit`, `tcgen05.wait_load/store` | [instructions/tcgen05.py](../../python/tilus/lang/instructions/tcgen05.py) |
| Hopper MMA | `wgmma.fence/mma/commit_group/wait_group` | [instructions/wgmma.py](../../python/tilus/lang/instructions/wgmma.py) |
| Async barriers | `mbarrier.alloc/arrive/arrive_and_expect_tx/wait`, with `scope='cta'\|'cluster'`, `sem='release'\|'acquire'\|'relaxed'` | [instructions/mbarrier.py](../../python/tilus/lang/instructions/mbarrier.py) |
| Proxy fences | `fence.proxy_async(space='shared')`, `fence.proxy_async_release()` | [instructions/fence.py](../../python/tilus/lang/instructions/fence.py) |
| Cluster | `cluster.sync()`, `blockIdx`, `clusterDim`, `blockRank`, `map_shared_addr(addr, rank)` | [instructions/cluster.py:42-138](../../python/tilus/lang/instructions/cluster.py#L42-L138) |
| Dtype reinterpret | `view(x, dtype=uint32)` on a register tensor — monotonic float→uint bit-flip is expressible as `(x & 0x80000000) ? ~x : (x \| 0x80000000)` once we have the uint32 register view | root.py (view op) |
| Reduce along a register-tensor dim | `sum`, `max`, `min` with `dim=` — per-thread reduction (fine for the 64-head weighted sum) | root.py (862-1607) |
| Printf / print_tensor | Useful for debugging cache indices | root.py |

### 1.3 The existing examples tell us how to structure each phase

- **FP8 paged gather + dequant** — [examples/quantization/matmul_a16wx.py](../quantization/matmul_a16wx.py) is the cleanest reference for "load uint8, bit-manipulate with `view()`, multiply by a scale". The indexer's `[num_pages, 64, 1, 132]` → `[num_pages, 64, 128] fp8 + [num_pages, 64] fp32` split is a 1:1 application of the same pattern.
- **Warp-specialized TMA + MMA + mbarrier pipeline** — [examples/hopper_matmul/matmul_v3.py](../hopper_matmul/matmul_v3.py) is the closest template for the logits kernel, just with `wgmma` instead of `tcgen05` for SM90. For SM100, [examples/blackwell_matmul/matmul_v1.py](../blackwell_matmul/matmul_v1.py) is the right template.
- **Online softmax-style reductions with scalar `.item()` reads** — [examples/flash_attention_decode/tilus_kernel.py](../flash_attention_decode/tilus_kernel.py) shows how per-row reductions and scalar thresholds flow in Tilus today.

---

## 2. Proposed Tilus structure

### 2.1 Kernel A — `dsv3_indexer_logits` (fused logits + optional histogram)

This is the "feasible today, modulo histogram fusion" kernel. Model it after [blackwell_matmul/matmul_v1.py](../blackwell_matmul/matmul_v1.py) with the following shape mapping:

| Matmul role | Indexer role |
|---|---|
| `A ∈ [BM, BK]` | K tile `[BL, 128]` where `BL = 64` (one page) or `128` (two pages) |
| `B ∈ [BN, BK]` | Q `[64, 128]` — shared by all tiles for a given batch item |
| `D ∈ [BM, BN]` in TMEM | Logits tile `[BL, 64]` in TMEM (relu + weighted reduce applied in epilogue to shape `[BL]`) |

Key steps (happy-path SM100 design, 1 warpgroup producer + 1 MMA warp + epilogue warps):

```python
@tilus.autotune("BL", [64, 128])
@tilus.autotune("num_epi_warpgrps", [1, 2])
class DSV3IndexerLogits(tilus.Script):
    def __call__(self, q_ptr, k_cache_ptr, w_ptr, seq_lens_ptr, block_table_ptr,
                 logits_ptr, histogram_ptr, batch_size, max_model_len, max_pages):
        self.attrs.blocks = [cdiv(max_pages, BL // 64), batch_size]   # 1 row of pages per block
        self.attrs.warps  = 2 * num_epi_warpgrps + 1 + 1               # epi + MMA + LDG + LDG-scales

        # 1. TMA Q once into shared, broadcast to TMEM half (or wgmma Bdesc).
        # 2. For page in pages_for_this_block:
        #    - TMA K fp8 page into s_k[stage]
        #    - copy_async scales into s_scale[stage]  (only 64 × 4B — cp.async is fine)
        #    - mbarrier.arrive_and_expect_tx
        # 3. MMA warp: tcgen05.mma(s_k, s_q, t_acc, enable_input_d=False) per k-chunk
        # 4. Epilogue warps:
        #    - wait mma mbarrier
        #    - tcgen05.load t_acc -> r_acc [BL, 64]
        #    - r_acc = self.maximum(r_acc, 0.0)       # relu
        #    - r_acc = r_acc * s_scale[broadcast]     # per-token dequant scale
        #    - r_logit = self.sum(r_acc * r_weights, dim=1)   # [BL]
        #    - self.store_global(g_logits, r_logit, offsets=[b, page*64])
        #    - *** HISTOGRAM FUSION: atomic_add(shared_hist[bin], 1) per element ***
        # 5. Block-wide: atomic_add shared_hist to global histogram
```

Everything above except the two lines marked `***` is already writable with the current instruction set. See §4.1 for the histogram gap.

### 2.2 Kernel B — `dsv3_topk` (radix partition-based top-K)

Single CTA per batch item (or a cluster of 2–8 CTAs per batch item for short L). Within a CTA:

```python
class DSV3TopK(tilus.Script):
    def __call__(self, logits_ptr, seq_lens_ptr, hist_ptr, indices_ptr, batch_size, max_model_len):
        self.attrs.blocks = [num_clusters, batch_size]
        self.attrs.cluster_blocks = [num_clusters, 1, 1]
        self.attrs.warps  = 32                                            # 1024 threads

        # s_hist[2][256]     two histogram buffers, ping-ponged between passes
        # s_cache_bits[2 * num_cached]  candidate values (uint32 bits)
        # s_cache_idx [2 * num_cached]  candidate indices
        # s_topk_idx[TOPK]              final output buffer
        # s_count, s_final_count        scalar counters (shared)

        # ---- Phase 0: build or load initial histogram ----
        if PRE_HISTOGRAM:
            # Parallel copy global hist_ptr[b] -> s_hist[0]
        else:
            for i in self.range(tid, L, NCLUSTERS * 1024):
                v = global_logit[i]
                bin = view(v, uint32).__apply_monotonic() >> 24
                atomic_add(s_hist[0][bin], 1)          # *** shared atomic ***

        # ---- Phase 0 threshold ----
        tau = cum_sum_top_down(s_hist[0], TOPK_REMAINING)   # needs warp scan + cluster broadcast

        # ---- Phase 0 partition ----
        for i in self.range(tid, L, NCLUSTERS * 1024):
            v = global_logit[i]
            bits = view(v, uint32).__apply_monotonic()
            bin = bits >> 24
            if bin > tau:
                off = atomic_add(s_final_count, 1)       # *** shared atomic ***
                if off < TOPK: s_topk_idx[off] = i       # *** dynamic-offset shared store ***
            elif bin == tau:
                off = atomic_add(s_count[0], 1)          # *** shared atomic ***
                if off < num_cached:
                    s_cache_bits[off] = bits
                    s_cache_idx [off] = i
                    atomic_add(s_hist[1][(bits >> 16) & 0xff], 1)

        # ---- Phases 1-3: same idea but walking s_cache not global_logit ----
        # Each pass extracts byte t, builds/reads s_hist[t%2], partitions.

        # ---- Finalize: CTAs with rank > 0 atomic_add into CTA-0's final_count
        #      to get a write offset, then copy s_topk_idx -> global indices[b, off:off+n]
```

Every `***`-marked line is a feature Tilus doesn't expose today.

---

## 3. Primitive-by-primitive mapping

| PTX / CUDA primitive used by reference | Tilus today? | Notes |
|---|---|---|
| `cp.async.bulk.tensor.3d` (Q / K loads) | Yes | `tma.global_to_shared` with TMA descriptor |
| `cp.async.bulk` 1D (scale bytes) | Yes | Same, 1D variant; or plain `copy_async` for SM80 |
| `tcgen05.mma.kind::f8f6f4` | Yes | `tcgen05.mma`, FP8 operands |
| `tcgen05.ld.sync.aligned.32x32b.xN` | Yes | `tcgen05.load` + `tcgen05.wait_load` |
| `tcgen05.commit.mbarrier::arrive::one` | Yes | `tcgen05.commit(mbarrier=...)` |
| `mbarrier.init / arrive / try_wait / arrive.expect_tx` | Yes | Full coverage in `mbarrier` module |
| `fence.mbarrier_init.release.cluster` | Yes | Emitted automatically when `scope='cluster'` on arrives |
| `fence.proxy.async.shared::cta` | Yes | `fence.proxy_async(space="shared")` |
| `elect.sync` / warp-elect | Yes | Covered implicitly by `single_thread()` |
| `ld.global.nc.L1::no_allocate.L2::256B.v2.f32` | Partial | `load_global` generates LDG; the exact cache hints are not user-tunable today. Functionally equivalent; perf may differ. |
| `__ffma2_rn` (packed FMA) | Partial | Tilus compiles `r * w` element-wise and will likely emit plain FFMA; the packed `ffma2` optimization is a backend concern. Minor perf gap. |
| `__float_as_uint`, monotonic bit-flip | Yes | Use `view(r_logit, dtype=uint32)` then arithmetic on uint32. |
| Extract radix byte `(bits >> 24-8t) & 0xff` | Yes | Plain arithmetic on uint32 register tensor. |
| `atomicAdd(shared_hist[bin], 1)` | No | **Not exposed as a Tilus instruction.** Hidet has `atomic_add` but Tilus' `root.py` does not wrap it. See §4.1. |
| `atomicAdd(&s_cached_count, 1)` returning the old value | No | Same blocker. Also: atomicAdd-returns-old semantics need to be a first-class op. |
| `atomicAdd(global_histogram + bin, local_hist[bin])` | No | Same. Global atomic also missing from the Script API. |
| `atomicAdd(cluster.map_shared_rank(&s_final_count, 0), n)` | No (double gap) | No shared atomic AND no "perform op on remote-CTA shared address". |
| `__shfl_sync`, `__shfl_xor_sync`, `__shfl_down_sync` | No | Exists in hidet (`elect.py:shfl_sync_i32`) but not surfaced as a Script instruction. Needed for cross-thread cum-sum inside the radix top-K pass. |
| `__ballot_sync`, `__activemask` | No | Not exposed. The reference uses them to find an "elected" writer. Could be worked around with warp id math. |
| `lanemask_lt / le / gt / ge`, `%laneid`, `%warpid` | No | None of these introspection intrinsics are Tilus Script primitives. |
| `cg::reduce`, `cg::exclusive_scan`, `cg::tiled_partition` | No | Cooperative-groups helpers used by `indexer_metadata.cu`. No Tilus equivalent. |
| `cluster.barrier_arrive / barrier_wait` (named cluster barriers) | Partial | Tilus has only `cluster.sync()` (full barrier). The reference uses half-barriers to overlap hist-clear with next-phase threshold compute. Can be emulated with `cluster.sync()` at some perf cost. |
| `cluster.map_shared_rank(addr, rank)` as an **address mapper** | Yes | `cluster.map_shared_addr(...)` returns a uint32 remote-shared address in a RegisterTensor. |
| **Actually reading / writing / atomic-add at that remote address** | No | Critical gap. You get the address but no instruction consumes it except `mbarrier.arrive(scope='cluster')`. No `remote_shared_atomic_add`, no `remote_shared_load`, no `remote_shared_store`. |
| `cudaTriggerProgrammaticLaunchCompletion()` / `cudaGridDependencySynchronize()` (PDL) | No | No PDL exposure. Has ~5–20% perf impact but not correctness-critical. |
| Dynamic-offset store into a register-computed shared address (`s_topk_idx[runtime_offset] = i`) | No | `store_shared(dst, src, offsets=Sequence[int])` only accepts compile-time `int` offsets, not `Expr`. See [root.py:594-635](../../python/tilus/lang/instructions/root.py#L594-L635). |

---

## 4. The blocker list — what Tilus needs to add

Ordered roughly by "how much of the indexer it unblocks":

### 4.1 Atomic tile operations (shared + global, element-wise + scatter)

Used by every part of the top-K kernel and the histogram fusion in the logits epilogue. Finalized API:

```python
# Element-wise atomic: dst and values shapes must match exactly (same RegisterLayout),
# no broadcast. Returns a register tensor of old values with dst's shape.
r_old = self.atomic.shared_add(
    dst,                     # SharedTensor
    values,                  # RegisterTensor | scalar Expr | int/float   (wrapped to a RegisterTensor)
    *, sem='relaxed', scope='cta' | 'cluster', output=None,
) -> RegisterTensor         # shape = dst.shape

# Scatter atomic, torch-style. indices.shape == values.shape strictly, with identical
# RegisterLayout. dst.shape[d] >= indices.shape[d] for d != dim; along dim the user is
# on the hook for in-range indices (no runtime bounds check).
r_old = self.atomic.shared_scatter_add(
    dst,                     # SharedTensor
    *, dim: int,             # compile-time scatter axis
    indices,                 # RegisterTensor, int dtype
    values,                  # RegisterTensor; same shape + layout as indices
    sem='relaxed', scope='cta' | 'cluster', output=None,
) -> RegisterTensor         # shape = indices.shape
```

Global variants have identical signatures with a `GlobalTensor` dst.

Full family (v1 dtype coverage in parens):
- `{shared,global}_{add, sub, min, max}` — `int32`, `float32`
- `{shared,global}_{and, or, xor, exch, cas}` — `int32`, `uint32`
- Scatter variant (`*_scatter_{add, sub, min, max, ...}`) for every op above.

Parameters:
- `sem ∈ {'relaxed', 'acquire', 'release', 'acq_rel'}` — mirrors PTX.
- `scope ∈ {'cta', 'cluster', 'gpu', 'sys'}` — mirrors PTX. Must be `'cluster'` when `dst` is the output of `self.cluster.map_shared(...)`; see §4.2. No verifier enforcement in v1 — the user's responsibility.
- `output` is optional. During build the frontend always allocates a RegisterTensor for it; a late DCE pass sets it to `None` when no downstream use, and codegen emits the destination-less PTX form. This needs one new DCE capability: "side-effecting instruction with a separably-removable register binding." Cleanest implementation is a per-instruction marker that DCE recognizes, rather than a per-atomic special case.

The underlying hidet primitives already exist ([hidet/ir/primitives/cuda/atomic.py:178](../../python/tilus/hidet/ir/primitives/cuda/atomic.py#L178)); the work is wrapping them as Tilus instructions with layout inference/validation and the DCE extension.

### 4.2 `cluster.map_shared` + remote-shared load/store/atomic

Concrete instruction on `self.cluster`, not lang sugar:

```python
class BlockClusterInstructionGroup:
    def map_shared(
        self, smem: SharedTensor, target_rank: Expr | int,
    ) -> SharedTensor: ...   # new IR identity; same shape/dtype/layout as smem
```

IR + codegen design:
- `MapSharedInst` takes `inputs=(smem,)`, produces an `output: SharedTensor` with a fresh object identity and an attribute `target_rank: Expr`.
- Since Tilus SharedTensors are identity-based (`eq=False`), the output is a distinct IR value — no changes to the `SharedTensor` class are needed.
- Layout inference: `output_layout = input_layout` (identity rule).
- Codegen emits `mapa.shared::cluster.u32 %tmp, %local_addr, %rank` and records `addr_of(output) = %tmp` in the existing smem-tensor→address bookkeeping map. All downstream shared-accepting instructions then transparently use the remote address.
- Every shared-addressing instruction (atomic, future `load_shared`/`store_shared` distributed variants) remains unaware of local vs. remote. It reads `addr_of(dst)` and emits the op qualified by `::cta` or `::cluster` per the user's `scope=` choice.
- `free_shared` on a `MapSharedInst` output is already blocked by Tilus' existing rule that only `self.shared_tensor(...)` allocations are freeable.

PTX forms:
- `mapa.shared::cluster.u32 rd, ra, rb` — translate CTA-local shared address `ra` to cluster-shared address at rank `rb`.
- Atomic RMW: `atom.{sem}.cluster.shared::cluster.add.T %old, [%mapped], %val` (and analogous `sub/min/max/and/or/xor/exch/cas`).
- Plain load/store (used for the cross-CTA histogram broadcast in the cluster top-K): `ld.{sem}.cluster.shared::cluster.T` and `st.{sem}.cluster.shared::cluster.T`.
- Constraint: when the address space is `.shared::cluster`, PTX requires `scope >= .cluster`. `.cta` with a mapped address is an assembler error.

### 4.3 Non-atomic tile scatter into shared / global

Used in the top-K for `s_topk_idx[off] = i` — each tile lane writes its value at a lane-specific offset. Parallel to `atomic.shared_scatter_add`, without the atomic RMW:

```python
self.store_shared_scatter(
    dst,                     # SharedTensor
    *, dim: int,
    indices,                 # RegisterTensor, int dtype
    values,                  # RegisterTensor; same shape + layout as indices
)

self.store_global_scatter(
    dst,                     # GlobalTensor
    *, dim: int,
    indices,                 # RegisterTensor, int dtype
    values,                  # RegisterTensor; same shape + layout as indices
)
```

Same shape/layout rule as the scatter atomic: indices and values share a RegisterLayout; dst has the torch relationship with indices. The scatter op is the tile-native way to express "each lane writes at a computed index" — we do not need to widen `store_shared(offsets=...)` to accept `Expr`.

Concurrent scatter with duplicate indices is UB by convention (last-writer-wins at the PTX level, but which one is unspecified). If correctness under duplicates matters, use `atomic.{shared,global}_scatter_{add, exch, ...}` instead.

### 4.4 Tile-level scan

The radix top-K's `cum_sum` (inclusive scan of the 256-bin histogram to find the threshold) is the one operation in this kernel that genuinely needs cross-thread movement beyond what atomics give us. Exposing it as a tile op rather than raw warp intrinsics keeps the kernel author in tile-space:

```python
r_cum = self.scan(
    x,                       # RegisterTensor
    *, op: str,              # 'add' | 'max' | 'min' | 'and' | 'or' | 'xor'
    dim: int,
    exclusive: bool = False,
) -> RegisterTensor         # same shape as x

# Shortcuts in the lang layer
r_cum = self.cumsum(x, dim=0)           # self.scan(x, op='add', dim=0)
r_cum = self.cumprod(x, dim=0)          # self.scan(x, op='mul', dim=0)
# (and cummax / cummin if useful)
```

Semantics match `torch.cumsum` / `torch.cummax` — inclusive by default, `exclusive=True` shifts the result by one position with the identity value at index 0.

Codegen strategy (follows the `ReduceInst` template in [backends/emitters/reduce.py](../../python/tilus/backends/emitters/reduce.py)):

- **Layout-driven dispatch.** The input's `RegisterLayout` tells the emitter how `dim` is distributed: purely local-to-thread, purely spread-across-lanes, or mixed.
- **Local-within-thread dim** → serial scan in registers, no cross-thread communication.
- **Warp-distributed dim** → Kogge-Stone / Brent-Kung warp scan built from `shfl_up_sync` (5 rounds for a 32-lane warp).
- **Block-distributed (cross-warp) dim** → warp-local scan → warp totals spill to shared → one warp-scan over the 32 totals → broadcast offsets back → add to each warp's local scan. Essentially the same pattern reduce already uses for cross-warp collapses.
- **Mixed layout** → compose the above phases.

Layout inference: scan preserves the input layout (`output_layout = input_layout`), unlike reduce which collapses along `dim`. Simpler than reduce's layout rules in [ir/layout/inference/inference_rules/reduce.py](../../python/tilus/ir/layout/inference/inference_rules/reduce.py).

This gives us what we need for the radix top-K (256-bin cumsum per pass, 4 passes) without any per-thread intrinsics in user code. If a future kernel genuinely needs raw `shfl`/`ballot`/`laneid`, we can add them later as an explicit escape hatch — the indexer doesn't.

### 4.5 Per-token scale dequant (cosmetic, not a blocker)

FP8 × per-token FP32 scale is writable today as `cast(r_fp8, f32) * r_scale` with a broadcast. The reference packs this into the load path via a custom TMA descriptor layout trick; Tilus doesn't need to match that — a plain register multiply in the epilogue is fine.

### 4.6 Nice-to-haves (perf only, not correctness)

- **Named cluster half-barriers** (`barrier.cluster.arrive/wait` separately) — the reference overlaps next-phase-histogram-clear with current-phase-threshold compute. With only `cluster.sync()`, we'll serialize; measurable but tolerable loss.
- **PDL** (`cudaTriggerProgrammaticLaunchCompletion`) — enables the top-K kernel to start while the logits kernel is still running.
- **Packed FMA (`ffma2`)** — backend optimization, invisible to Tilus user code.
- **Cache hint on `load_global`** — `L1::no_allocate.L2::256B` is used heavily in the radix-scan over billions of floats; for L ~ 500K × batch, this pays off. The ask is to let `load_global` / `copy_async` take a `cache_policy=...` hint.

### 4.7 Keep `indexer_metadata` out of Tilus

It's 104 lines of CG helpers, runs once per shape, and is dwarfed by the actual kernels. Recommended: keep it as an escape-hatch hidet/Python kernel or a CPU prep step. Don't block the Tilus port on it.

---

## 5. Recommended implementation plan

Decided policy: land the new Tilus instructions first, *before* starting the indexer kernel. No hidet escape hatches inside Tilus scripts.

### 5.1 PR order (instructions first)

| PR | Lands | Unblocks milestone |
|---|---|---|
| **PR 1** | `self.atomic.{shared,global}_{add, sub, min, max, and, or, xor, exch, cas}` element-wise. `sem`/`scope` params, optional `output`. Generic DCE capability for "side-effecting instruction with a removable register binding." | M3 — logits epilogue histogram fusion + block→global histogram flush. |
| **PR 2** | Scatter variants: `self.atomic.{shared,global}_scatter_{add, sub, min, max, and, or, xor, exch, cas}` with `dim` parameter, strict `indices.shape == values.shape`, identical RegisterLayout. | M4 — radix top-K histogram build (`atomic_add(s_hist[bin], 1)`). |
| **PR 3** | Non-atomic scatter stores: `self.store_shared_scatter` / `self.store_global_scatter`. | M4 — `s_topk_idx[off] = i` in the top-K commit path; M7 — global overflow spill. |
| **PR 4** | `self.cluster.map_shared(smem, target_rank)` as `MapSharedInst`. Codegen emits `mapa.shared::cluster`; all shared-accepting instructions transparently work on the output when the user sets `scope='cluster'`. Includes distributed `load_shared`/`store_shared` with `.shared::cluster` qualifier. | M6 — multi-CTA cluster variant of top-K (`NClusters > 1`). |
| **PR 5** | Tile-level scan: `self.scan(x, op, dim, exclusive)` with `cumsum`/`cumprod` shortcuts in the lang layer. Layout-driven codegen following the `ReduceInst` template in [backends/emitters/reduce.py](../../python/tilus/backends/emitters/reduce.py) — local-in-thread scan serialized, warp-distributed dim via `shfl_up_sync`, cross-warp via shared-memory fixup. Can land in parallel to PR 1–4. | M4 — 256-bin radix threshold search (one scan per pass, four passes). |

### 5.2 Kernel milestones (each independently testable against `baseline.py`)

| M | Kernel | Depends on |
|---|---|---|
| **M1** | Logits-only Tilus kernel (FP8 MMA + relu + per-head weighted sum + per-token scale). Output `logits[B, max_model_len]`. No histogram side effect. SM100 via `tcgen05`; SM90 fallback via `wgmma`. | Today's Tilus instruction set. |
| **M2** | Wire M1 into `benchmark.py` as the `tilus` column. Validate against baseline. | M1. |
| **M3** | Logits with fused histogram epilogue. | PR 1. |
| **M4** | Single-CTA fast (lossy) radix top-K, `PRE_HISTOGRAM=False`. Benchmark against `torch.ops._C.top_k_per_row_decode`. | PR 1, PR 2, PR 3 (+ PR 5 for perf). |
| **M5** | Top-K with `PRE_HISTOGRAM=True` reusing M3's histogram. | M3, M4. |
| **M6** | Cluster variant (`NClusters ∈ {2, 4, 8}`). | PR 4. |
| **M7** | `_exact` variant with global overflow spill. | M4, PR 3 (global scatter). |
| **M8** | Perf tuning: cache hints on `load_global`, named cluster half-barriers, PDL. | §4.6. |

M1–M2 can start today in parallel with PR 1–5; they'll be ready to compose as PRs land.

---

## 6. Open questions / things worth confirming before starting

1. **Target SM family.** DeepGEMM supports SM90 + SM100/103. Tilus examples have both `wgmma` and `tcgen05` templates. Is SM100 the primary target, or do we need both? This affects whether the logits kernel is one script with two autotuned compute backends or two scripts.
2. **Precision.** The reference accumulates logits in FP32 via `ffma2` after an FP8 → FP32 MMA. Is matching that good enough, or do we need to match FlashInfer's exact rounding on pathological inputs?
3. **Top-K semantics.** Does the end-to-end DSV3.2 decode run tolerate the `fast` (lossy) variant, or must we do `exact` from day one? (The vLLM column dispatches both via heuristic.)
4. **Instruction-set scope creep.** Adding user-facing atomics and shuffles is useful beyond this kernel; should the PRs that land them be standalone, or bundled with the indexer?

---

## 7. Summary

- **Logits kernel**: essentially writable with today's Tilus — mirror [blackwell_matmul/matmul_v1.py](../blackwell_matmul/matmul_v1.py), paged-K pattern from [attention_with_kvcache/attention_v1.py](../attention_with_kvcache/attention_v1.py). Histogram fusion needs PR 1.
- **Top-K kernel**: needs five instruction additions, finalized as tile-native ops:
  - **PR 1** element-wise atomics (`self.atomic.{shared,global}_*`)
  - **PR 2** scatter atomics (`self.atomic.*_scatter_*`, torch-style `dim`)
  - **PR 3** non-atomic scatter stores (`self.store_shared_scatter` / `self.store_global_scatter`)
  - **PR 4** `self.cluster.map_shared` + distributed shared load/store/atomic via `mapa.shared::cluster`
  - **PR 5** tile-level `self.scan(x, op, dim, exclusive)` with `cumsum`/`cumprod` shortcuts
- **Metadata kernel**: leave outside Tilus.

Policy: land the five PRs first, no hidet escape hatches in Tilus scripts. M1–M2 (logits-only) can develop in parallel and will compose as PR 1–5 land.

The logits half is the high-value, low-risk first milestone; the top-K half is the feature work that makes Tilus strictly-more-capable in a way that benefits other kernels too (sparse MoE, stream-K reductions, histogram-based ops).
