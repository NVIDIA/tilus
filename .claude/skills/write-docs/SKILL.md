---
name: write-docs
description: >
  Convention and format for writing instruction docstrings and RST tutorials in
  tilus documentation. TRIGGER when: user asks to add, update, or write
  documentation for tilus instructions, instruction groups, or tutorials.
---

# Writing Instruction Documentation

## Docstring Format

All instruction docstrings use **NumPy-style** format with the following structure:

```python
def method_name(self, param1: Type1, param2: Type2) -> ReturnType:
    """One-line summary of what the instruction does.

    Extended description explaining the behavior, semantics, and constraints
    of the instruction. This can be multiple paragraphs.

    Parameters
    ----------
    param1: Type1
        Description of the parameter. Include constraints and valid ranges
        (e.g., "must be evaluated to a positive int32").
    param2: Type2
        Description. For parameters with defaults, explain the default behavior
        (e.g., "By default, it is 1.").

    Returns
    -------
    ret: ReturnType
        Description of the return value including shape, dtype, and relationship
        to inputs.
    """
```

## Conventions

### Summary line
- Start with a verb: "Allocate...", "Arrive at...", "Compute the...", "Load...", "Wait at..."
- Keep it to one line

### Extended description
- Explain what the instruction does at the hardware level when relevant
- Document state transitions (e.g., barrier phase switching)
- Describe relationships between parameters
- Explain multicast/cluster behavior for distributed instructions
- Scale detail to complexity: simple methods (load, cast) need minimal description; complex methods (mbarrier.alloc, tma.global_to_shared) need thorough explanation

### Parameters section
- Document in the same order as the function signature
- Use full type annotations: `RegisterTensor`, `Expr | int`, `Optional[Type]`
- Include constraints: "must be in the range of [0, N)", "must be evaluated to a non-negative int32"
- Document valid candidates for string parameters: `Candidates: 'relaxed', 'release'.`
- Explain default values: "By default, it is 1."

### Returns section
- Use `ret: Type` format
- Describe shape, dtype, and semantic meaning

### Notes section (required)
Every instruction must have a Notes section with these items as a compact bullet list:

```python
        Notes
        -----
        - **Thread group**: Can be executed by any sized thread group.
        - **Hardware**: Requires compute capability 8.0+ (sm_80).
        - **PTX**: ``mbarrier.init.shared::cta.b64``
```

The three standard note items:

- **Thread group**: Execution requirements. Common values:
  - "Can be executed by any sized thread group."
  - "Must be executed by a warp group (4 warps)."
  - "Must be executed by a single thread (use ``self.single_thread()``)."
  - "Must be executed by a single warp (use ``self.single_warp()`)"
- **Hardware**: Minimum compute capability. Format: "Requires compute capability X.Y+ (sm_XY)."
  - sm_80 = Ampere (A100), sm_89 = Ada Lovelace (L4/L40), sm_90 = Hopper (H100), sm_100 = Blackwell (B200)
- **PTX**: The underlying PTX instruction(s) this maps to. Use double backticks for inline code.
  If the instruction maps to multiple PTX instructions depending on parameters, list them:
  ```
  - **PTX**: ``mbarrier.arrive.shared::cta.b64`` or ``mbarrier.arrive.noComplete.shared::cta.b64``
  ```
  If the instruction does not lower to a specific PTX instruction (e.g., it's a high-level
  construct), omit the PTX line.

### Other optional sections
- **Examples**: Use `.. code-block:: python` for usage examples
- **See Also**: Cross-reference related methods with `:py:meth:` or `:py:func:`

### Memory ordering parameters
For synchronization instructions, document `sem` and `scope` parameters consistently:
```python
sem: str
    The memory ordering semantics for the operation. Candidates: 'relaxed', 'release'.
scope: str
    The synchronization scope for the operation. Candidates: 'cta', 'cluster'.
```

## Reference examples
- Simple instruction: `root.py:load_global`, `root.py:cast`
- Complex instruction: `mbarrier.py:alloc`, `mbarrier.py:arrive_and_expect_tx`
- With PTX reference: `fence.py:proxy_async`, `fence.py:proxy_async_release`
- With code example: `root.py:range`, `root.py:thread_group`
- TMA instruction: `tma.py:global_to_shared`

---

# Writing RST Tutorials

## Target audience

Tutorials target **CS researchers who can write Triton kernels** but want to
understand the hardware features underneath Triton's abstractions. Assume readers
know:
- Block-level GPU programming (each program instance processes a tile)
- `tl.load`, `tl.store`, `tl.dot` semantics
- Autotuning concepts
- Basic GPU memory hierarchy (global, shared, registers)

Do **not** assume they know:
- Explicit shared memory management or allocation
- Warp-level programming or thread indexing within a block
- Asynchronous execution models (mbarrier, TMA, commit/wait patterns)
- Tensor Memory or tcgen05 instruction families
- Memory ordering semantics (acquire/release, proxy fences)

## Bridging the gap from Triton

When introducing a concept that Triton handles implicitly, briefly explain **why**
explicit control is needed. Common contrasts:

- **Shared memory**: "Unlike Triton where shared memory is managed automatically,
  tilus gives explicit control --- necessary to use hardware features like TMA and
  tcgen05."
- **Registers vs Tensor Memory**: "On earlier architectures (and in Triton),
  MMA results accumulate in registers. Blackwell's tensor cores use dedicated
  Tensor Memory, which provides higher bandwidth and avoids consuming register
  file capacity for large tiles."
- **Synchronization**: "Triton handles synchronization implicitly. On Blackwell,
  many operations are asynchronous --- the instruction returns immediately and
  completes in the background. This enables overlap of data movement and
  computation, but requires explicit tracking via mbarriers."
- **Thread/warp management**: "In Triton, all threads execute the same code.
  Efficient Blackwell kernels require different warps to perform different
  jobs (loading, computing, scheduling) and collaborate asynchronously via
  thread groups."

## Tutorial structure

Each tutorial version (v0, v1, ...) should follow this structure:

1. **Introduction** --- What this version adds, what Blackwell features it uses
   (with hyperlinks to instruction group docs).
2. **Full kernel** --- Show the complete kernel upfront so readers see the big
   picture before the detailed walkthrough.
3. **Topic sections** --- Explain each new concept with enough detail to
   understand the example. Order by conceptual dependency. Include:
   - A brief motivation (why does this exist / why do we need it)
   - How it works at a high level
   - Link to the detailed API/programming guide for deeper reading
4. **Walkthrough** --- Walk through the kernel code in logical groups (setup,
   main loop, epilogue). Use `literalinclude` with `:start-at:`/`:end-at:`
   markers (never absolute line numbers). For each group, use a bullet list
   explaining each instruction with hyperlinks.
5. **What's Next** --- Motivate the next version by identifying the current
   bottleneck.
6. **Full Source** --- Download link to the example file.

## Writing guidelines

### Explain the "why", not just the "what"
- For every `sync()` call, explain what it guards (e.g., "ensures shared memory
  writes are visible to all threads before the MMA warp reads them").
- For magic numbers like `warps = 4`, explain the choice (e.g., "4 warps = 128
  threads; later versions use more warps to overlap loading and computing").
- For `enable_input_d`, explain: "On the first iteration, tensor memory contains
  uninitialized data, so we ignore it. On subsequent iterations, it holds the
  running sum from prior tiles."
- For mbarrier phase flipping, explain: "The same barrier is reused across
  iterations. The phase distinguishes this iteration's completion from the
  previous one's."

### Hyperlinks
- Use `:meth:` for instruction methods:
  `:meth:`~tilus.Script.copy_async`` for root instructions,
  `:meth:`tcgen05.mma <tilus.lang.instructions.tcgen05.Tcgen05InstructionGroup.mma>``
  for instruction group methods (shows short name, links to full path).
- Use `:attr:` for attributes: `:attr:`self.attrs.blocks <tilus.lang.script.Attributes.blocks>``
- Use `:doc:` for cross-references to other pages: `:doc:`/programming-guides/thread-group``
- Use `:class:` for tensor types: `:class:`~tilus.ir.tensor.TMemoryTensor``

### Code inclusion
- Always use `:start-at:` / `:end-at:` / `:start-after:` / `:end-before:`
  instead of absolute line numbers. This makes includes resilient to code
  changes.
- Use `:dedent:` to strip leading indentation when including method bodies.
- Use `:caption:` for all included code blocks.

### Figures and diagrams
- Place SVGs in a `figures/` subdirectory next to the tutorial RST files.
- Use `.. figure::` with `:width:` and `:align: center`.
- SVGs should be editable in draw.io for collaborative iteration.
- Suggest diagrams for: block tiling, data flow, pipeline stages,
  cluster layouts, and any concept that benefits from a visual.

### Tone
- Concise and direct. Avoid filler words.
- Vary sentence structure --- avoid starting every bullet with "We ..."
  (lead with the instruction/concept name instead).
- Don't over-explain concepts that are well-covered in linked pages. The tutorial
  should give enough to understand the example; detailed semantics belong in the
  programming guides and API docs.

## Reference tutorial
- Blackwell matmul V0: `docs/source/tutorials/matmul-blackwell/v0.rst`
