## Hidet Integration Refactor — Context for Continuation

### Goal
Refactor tilus to directly integrate the needed parts of hidet (IR, passes, runtime, lang, backend) into `tilus.hidet`, eliminating the external `hidet` package dependency. Both repos are NVIDIA-owned by the same author.

### What's Been Done

**Phase 0-1: Copy hidet source into `tilus.hidet`**
- Created `python/tilus/hidet/` package with submodules: `ir/`, `transforms/`, `lang/`, `backend/`, `drivers/`, `ffi/`, `utils/`
- Copied hidet source files from the hidet repo (tag matching current dependency)
- Rewrote all internal imports from `hidet.X` to `tilus.hidet.X`
- Fixed circular imports (e.g., `ir/tools/__init__.py` → `ir/utils/call_graph.py` cycle)

**Phase 2: Merge `tilus/extensions/hidet/` customizations into `tilus.hidet`**
Six parallel agents handled:
1. **dtypes** — Added `FloatSubbyteType`, sub-byte float/int types, vector types (`uint32x1/x2/x4`), `mantissa_nbits`/`exponent_nbits` attributes
2. **primitives** — 16 new CUDA primitive files (tcgen05, mbarrier, fence, copy_async_bulk/tensor, cast, clc, elect, etc.) + extensions to existing ones (ldst, cp_async, wgmma, cluster, vars, tensor_map, etc.)
3. **IR tools/builders/utils** — Extended `type_infer.py`, `simplifier.py`, added `verifier.py`, `TypedStmtBuilder`, extended `expr.py`/`module.py`/`type.py`/`index_transform.py`
4. **transforms** — 6 new passes (deadcode_elimination, bind_predefined_variables, hoist_loop_invariants, lower_affine_to_recurrence, lower_subbyte_type, lower_float8_cast) + 5 overridden passes
5. **backend + lang** — `UpdatedCUDACodeGen` merged into `codegen.py`, `build.py` created, transpiler extended with `Starred` unpacking support
6. **libinfo + utils** — `find_include_path()`, include headers, hidet utils ported to `tilus/utils/` (prod, gcd, lcm, Doc, etc.), `tilus/hidet/utils/` shim package

**Phase 3: Update all tilus-side imports**
- Ran `rewrite_imports.py` to change all `from hidet.X` and `from tilus.extensions.hidet.X` to `from tilus.hidet.X` across ~80 files in `python/tilus/`
- Updated examples (`attention/`, `quantization/`) and `tests/conftest.py`

**Bug fixes applied:**
- `cp_async.py` — Accept `Constant` exprs for `allow_on_fly_groups` in `cp_async_wait_group`
- `type.py` — Added `OpaqueType` and `ReferenceType` handling in `type_equal()`
- `module.py` — Implemented `IRModule.build()` using `tilus.drivers.build_ir_module` + `CompiledModule`
- `compiled_program.py` — Added `CompiledModule` class (runtime wrapper for `IRModule.build()`)
- `namer.py` — Fixed `isinstance()` crash when `ScalarNode`/`TensorNode` are `None`
- `test_cast.py` — Switched from `hidet.script` to `tilus.hidet.lang.script`
- Various circular import fixes in `transforms/__init__.py`, `lang/__init__.py`, primitives

### What Remains

**Phase 4: Cleanup**
- Delete `python/tilus/extensions/hidet/` (the old extension layer, now merged)
- Remove all remaining references to `tilus.extensions.hidet` in imports
- Update `pyproject.toml` to remove `hidet` from dependencies (or keep as optional for now)
- Clean up any `import hidet` that should be `import tilus.hidet`

**Phase 5: Verification**
- Run `pre-commit run --all-files` (was passing as of last commit)
- Run `tests/kernels/matmul/test_matmul_v2.py` as smoke test (per CLAUDE.md guidance)
- Run full test suite, fix remaining failures
- Known pre-existing failures (NOT caused by refactor):
  - `matmul_v7` — CUDA launch failure during autotuning (same on main)
  - `matmul_v8` — also fails on main (namer.py fix applied but may have other issues)

**Remaining stale imports to watch for:**
- `python/tilus/hidet/ffi/ffi.py` still has `import hidet.ffi.ffi` (runtime FFI bridge)
- `python/tilus/hidet/libinfo.py` still has `import hidet.libinfo` (fallback for include paths)
- `python/tilus/hidet/backend/build.py` may still reference `hidet.cuda` and `hidet.option`
- Any test files under `tests/extensions/` that still use `@hidet.script` or `hidet.script_module()`

### Key Design Decisions
1. Module path: `tilus.hidet` (not `tilus._hidet` or `tilus.ir`)
2. Extensions merged into the copied code (not kept separate)
3. Still depends on `torch` for tensor operations
4. `tilus.hidet.utils/` is a shim that re-exports from `tilus.utils/` where possible
5. `IRModule.build()` uses `tilus.drivers.build_ir_module` → `CompiledModule` (in `tilus.runtime`)

### Branch
`refactor-hidet-integration` — latest commit: `b9f0b19` "fix: resolve test failures from hidet integration refactor"

### File counts
- ~170 files copied/created under `python/tilus/hidet/`
- ~80 files had imports rewritten in `python/tilus/`
- ~10 files in `examples/` and `tests/` updated

### How to test after changes
```bash
# Smoke test (fast)
python -m pytest tests/kernels/matmul/test_matmul_v2.py -x -q

# Full suite (slow)
python -m pytest tests/ --ignore=tests/extensions -x -q

# Pre-commit
pre-commit run --all-files
```
