---
name: write-docs
description: >
  Convention and format for writing instruction docstrings in
  python/tilus/lang/instructions/. TRIGGER when: user asks to add, update, or
  write documentation for tilus instructions or instruction groups.
  DO NOT TRIGGER when: user is working on Sphinx RST files or non-instruction docs.
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
