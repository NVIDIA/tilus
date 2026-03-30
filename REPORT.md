# Debugging Report: R16x256B + stmatrix

## Key Findings

### 1. Column Permutation Pattern
With `randn` input at 256×256×64, the matmul_v9 output with R16x256B has a **pure column permutation** — all values match the reference exactly but at wrong column positions:
```
c_r16[0, 0:8]   → c_ref[0, 8:16]   (cols swapped by groups of 8)
c_r16[0, 8:16]  → c_ref[0, 0:8]
c_r16[0, 16:24] → c_ref[0, 24:32]
c_r16[0, 24:32] → c_ref[0, 16:24]
```
The permutation matches `col_group XOR ((row >> 1) & 3)` — the swizzle XOR pattern.

### 2. Ones Input Passes, Randn Fails
`torch.ones` input produces correct output because the column permutation is invisible with uniform data. `torch.randn` reveals the permutation.

### 3. stmatrix + Swizzle Incompatibility
stmatrix writes 16 bytes sequentially from a given address. With swizzled shared layouts, the 4 column tiles of a row are NOT at consecutive 16-byte positions — they're permuted by the swizzle. stmatrix writes them sequentially, placing tiles at wrong column positions.

**However**, this is a SECONDARY issue. Disabling stmatrix (using generic stores) also fails for the matmul_v9 case. The swizzled generic store should handle element-by-element addressing correctly.

### 4. The PRIMARY Issue: tcgen05.ld R16x256B After MMA
The isolated test (tcgen05.store → tmem → tcgen05.ld R16x256B → store_shared → TMA) passes correctly.
The matmul_v9 (MMA → tmem → tcgen05.ld R16x256B → store_shared → TMA) fails.

The ONLY difference is how tmem data is written:
- `tcgen05.store.16x256b`: writes with R16x256B format → reads back correctly
- `tcgen05.mma`: writes in MMA format → R16x256B read produces wrong register content

Both R32x32B and R16x256B read the same tmem lanes/columns, but the register assignment differs. The MMA output in tmem might have a specific lane-to-column mapping that's compatible with R32x32B but not R16x256B.

### 5. stmatrix Address Fix Needed
Even when R16x256B works correctly, stmatrix cannot be used with swizzled shared layouts. Two options:
1. Infer non-swizzled shared layout for stmatrix path (and use TMA SWIZZLE_NONE)
2. Compute stmatrix addresses compatible with the specific swizzle mode

Currently the emitter skips stmatrix when the shared layout is swizzled.

## Next Steps
1. Investigate why `tcgen05.ld.16x256b` reads wrong data from MMA-written tmem
2. Either fix the R16x256B read to be compatible with MMA output, or ensure the layout inference produces R16x256B layout only when it matches the tmem data format
3. Resolve the stmatrix + swizzle interaction
