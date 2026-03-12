## Cache

Tilus caches generated kernels. During development, set the cache directory via `tilus.option.cache_dir("some-cache-dir")` to inspect the generated `source.cu` files.

### Cache structure
- **`/scripts/`** — Each script is a template with a tuning space. Contains references to its instantiated programs.
- **`/programs/`** — Each program is a concrete instantiation of a script with a specific schedule.

### Tips
- The cache directory can be safely deleted at any time.
- To inspect the CUDA kernel generated for a specific script, delete the cache directory, run the program, and check the newly generated `source.cu`.
- Use `debug_schedule=dict(...)` to pin a specific schedule, so only that single configuration is compiled.

## Compilation Pipeline

`drivers.py:build_program` orchestrates the full compilation:
1. **Verify** — `ir.tools.verify(prog)`
2. **Optimize (Tilus IR)** — `optimize_program` applies `get_default_passes()` from `transforms/__init__.py`
3. **Lower to Hidet IR** — `backends.codegen.generate_ir_module`
4. **Optimize (Hidet IR)** — `optimize_ir_module` applies Hidet-level passes
5. **Codegen** — Emit CUDA C source
6. **Compile** — Build `.so` via nvcc

### Tilus IR Structure

- **Program** (`ir/prog.py`) — `frozendict[str, Function]`
- **Function** (`ir/func.py`) — `name`, `params`, `body: Stmt`, `metadata: Metadata`
- **Stmt** (`ir/stmt.py`) — Frozen dataclasses: `SeqStmt`, `ForStmt`, `IfStmt`, `WhileStmt`, `LetStmt`, `InstStmt`, `ThreadGroupStmt`, etc.
- **InstStmt** wraps an `Instruction` — this is how instructions appear in the statement tree
- **Instruction** (`ir/inst.py`) — `output: Optional[Tensor]`, `inputs: tuple[Tensor, ...]`, plus type-specific attributes
  - **Functional instructions** — pure computations on tensors (e.g., `AddInst`, `CastInst`, `SliceRegisterInst`). Whether an instruction is functional is determined by an explicit allowlist, NOT by checking `output is None`.
  - **Side-effecting instructions** — memory ops, synchronization, etc. (e.g., `StoreGlobalInst`, `SyncThreadsInst`). Must never be eliminated.
- **Tensor** — `RegisterTensor`, `SharedTensor`, `GlobalTensor`, `TMemoryTensor` — identity-based (frozen dataclass with `eq=False`)

### Writing Passes

- Base class: `transforms/base.py:Pass` — override `process_function(func) -> Function`
- `IRRewriter` (`ir/functors/functor.py`) — visitor/rewriter pattern. Override `visit_*` methods. Uses identity-based memoization.
  - `visit_Instruction` handles all instruction types generically (rewrites output/inputs/attributes)
  - `visit_InstStmt` delegates to `visit(stmt.inst)` — if the instruction visitor returns `None`, the stmt becomes `SeqStmt(())`
  - For instruction-specific handling, define `visit_<InstructionClassName>` methods
- `IRVisitor` — read-only traversal (same dispatch, returns None)
- Register pass in `transforms/__init__.py:get_default_passes()` and export from `__init__.py`
- Tensors use identity (`is`) for equality since `eq=False`. The memo dict in IRFunctor keys on object identity for IR nodes.

### IRVisitor/IRRewriter Pitfalls

- **`visit_Expr` is a no-op in `IRVisitor`** — it does NOT descend into Hidet sub-expressions. To collect Vars from Hidet expressions, use `hidet.ir.tools.collect(expr, Var)` explicitly.
- **Memo prevents re-visiting** — once a node is visited, `IRFunctor.visit()` returns the cached result. If you need to process the same Expr from multiple contexts, don't rely on `visit_Expr` being called again. Instead, collect data directly (e.g., iterate `inst.attributes.values()` and call `hidet_collect` yourself).
- **Instruction attributes contain Hidet Exprs** — many instructions store Vars (e.g., barrier addresses, offsets) in dataclass fields beyond `output`/`inputs`. Access these via `inst.attributes` (a dict of all non-output/inputs fields). When analyzing Var usage, always scan attributes too.
- **`TensorItemValueStmt`/`TensorItemPtrStmt`** bind Hidet `Var`s to tensor values/pointers. They bridge the Tilus tensor world and the Hidet scalar expression world. When checking liveness, both the tensor and the bound Var must be considered.

### Testing Passes

- Build test IR directly using `Function.create(...)`, `SeqStmt(...)`, `InstStmt(inst)`, and instruction `create()` methods. See `tests/transforms/test_dead_code_elimination.py` for examples.
- Use `ir/tools/instruction_collector.py:collect_instructions(func)` to count instructions by type after a pass.
- For end-to-end testing, use `InstantiatedScript._jit_instance_for(...)` to get a `JitInstance`, then access `ji.transpiled_programs[0]` for the `Program`.

## Debug Tips

### Dumping IR after each pass

Call `tilus.option.debug.dump_ir()` before running the kernel. The IR after each pass will be dumped into the cache directory under `ir/` (for Tilus IR passes) and `module/ir/` (for Hidet IR passes).

### Proxy fence required between `store_shared` and `tma.shared_to_global`

`self.store_shared(...)` writes to shared memory via the **generic proxy**, while `self.tma.shared_to_global(...)` reads from shared memory via the **async proxy**. A `fence.proxy.async.shared::cta` is required between them to ensure the generic proxy writes are visible to the async proxy. Without this fence, the TMA engine may read stale data.

```python
    ...
    self.store_shared(s_c, ...)
    self.fence.async_view(space="shared")   # fence.proxy.async.shared::cta
    self.sync()
    with self.single_thread():
        self.tma.shared_to_global(s_c, g_c, ...)
        self.tma.commit_group()
        self.tma.wait_group(n=0)
    ...
```
