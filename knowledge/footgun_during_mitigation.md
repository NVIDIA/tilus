# TVM-FFI py_class Footguns Encountered During Loom Weave IR Migration

This document records the practical pitfalls hit while migrating Loom's Weave IR
(`@dataclass(frozen=True)` → `@tvm_ffi.dataclasses.py_class`). Verified against
`apache-tvm-ffi` from the `apache/tvm-ffi` repo, commit `02e9928a`.

## Field type system: not as flexible as Python typing

### Plain Python `Enum` is NOT accepted as a field type
```python
class Color(Enum):
    RED = 1
@py_class
class Foo(tvm_ffi.Object):
    color: Color   # ← TypeError: Cannot convert <enum 'Color'> to TypeSchema
```
**Workaround.** Annotate the field as `str` and make the enum a `StrEnum` so
its values are themselves strings (`isinstance(Color.RED, str) is True`).
The enum value is then stored as a plain `str` and `op.color == "red"` works.
Use `IntEnum` for enums whose values are integers (and field type `int`).

### Plain `dataclass` types are NOT accepted as field types
```python
@dataclass(frozen=True)
class Foo: ...
@py_class
class Container(tvm_ffi.Object):
    foo: Foo  # ← TypeError: Cannot convert <class 'Foo'> to TypeSchema
```
**Workaround.** Migrate the referenced dataclass to `py_class` too.

### Union types involving non-py_class types fail
```python
@py_class
class Foo(tvm_ffi.Object):
    field: SomeDataclass | int | None = None  # ← FAILS
```
**Workaround.** Migrate any dataclass referenced in unions to `py_class`.
Unions of primitives + py_class types work fine: `Expr | int | None`.

### `Any` works as an escape hatch but loses type info
A field annotated `Any` accepts arbitrary Python objects (stored as
`OpaquePyObject`). Useful as a transitional escape hatch but values become
opaque to the FFI layer (no structural eq/hash, no auto-print). The user
specifically forbade this in the loom migration.

## Container types: `tuple` ↔ `ffi.Array`, `list` ↔ `ffi.List`

py_class transparently rewrites `tuple[T, ...]` fields to `ffi.Array` and
`list[T]` to `ffi.List`. The values still iterate, index, and len() like
Python sequences, but:
- `isinstance(x, tuple)` → **False**
- `isinstance(x, list)` → **False**

**Workaround.** Use duck typing or `type(x).__name__ in ("Array", "List")` for
container detection. We added a small `is_seq()` helper:
```python
def is_seq(value):
    return isinstance(value, (tuple, list)) or type(value).__name__ in ("Array", "List")
```

## `cached_property` is incompatible with py_class

py_class uses `__slots__` and stores attributes in the C handle, not in
`__dict__`. `functools.cached_property` requires `__dict__` write access:
```python
@py_class
class Foo(tvm_ffi.Object):
    x: int
    @cached_property
    def y(self):
        return self.x * 100   # ← TypeError: No '__dict__' attribute
```
**Workaround.** Convert to plain `@property` (recompute each call) or maintain
your own cache in a side dict keyed by `obj.__chandle__()`.

## Identity is the C-handle, not the Python wrapper

Two different Python wrappers can wrap the same underlying C handle:
- `obj1 is obj2` → unreliable (Python identity)
- `id(obj1) == id(obj2)` → unreliable (Python id)
- `obj1.same_as(obj2)` → reliable
- `obj1.__chandle__() == obj2.__chandle__()` → reliable

**Implication.** Code that uses dict keys or `id()`-based memoization on IR
nodes must switch to handle-based identity. For `__hash__`, use
`return self.__chandle__()`.

## `frozen=True` blocks ALL mutation including `__post_init__`

```python
@py_class(frozen=True)
class Foo(tvm_ffi.Object):
    x: int
    y: int | None = None
    def __post_init__(self):
        if self.y is None:
            self.y = self.x * 2  # ← AttributeError: property has no setter
```
**Workaround.** Omit `frozen=True` when `__post_init__` mutates fields. Rely on
the dataclass-style convention of "don't mutate after construction" rather than
runtime enforcement.

## `__post_init__` IS supported (good news)

When NOT using `frozen=True`, py_class calls `__post_init__()` after
`__init__()` runs, just like stdlib dataclasses. Direct field assignment
(`self.y = inferred`) works without needing `object.__setattr__`.

## Custom `__eq__` returning non-bool works

py_class with default `eq=False` doesn't override Python's natural
`__eq__` mechanism, so DSL-style operator overloading (e.g. `Expr.__eq__`
returning a `Compare` AST node instead of `bool`) is preserved.

```python
@py_class
class Expr(tvm_ffi.Object):
    def __eq__(self, other):
        return Compare("==", self, other)   # works
```

## `isinstance()` works for inheritance

py_class registers parent types as ancestors, so `isinstance(child, base)`
returns True both ways: Python class hierarchy + FFI type registration.

## `ClassVar` annotations are correctly skipped

Class-level constants annotated with `ClassVar[...]` are not registered as
FFI fields and remain accessible as plain class attributes:
```python
@py_class
class Foo(tvm_ffi.Object):
    _SENTINEL: ClassVar[tuple[str, ...]] = ("a", "b")  # OK
    x: int                                              # field
```

## Defaults on py_class fields are not introspectable from Python

`inspect.signature(cls)` returns `ffi.Object` placeholder values for default
parameters. There's no Python-side API to read the actual default value of a
py_class field. Tools that compare field values to defaults (e.g. for
output-omission optimization) cannot do so for py_class objects.

## Subclass MUST inherit from a registered FFI type

```python
@py_class
class Foo:
    x: int    # ← TypeError: Foo must inherit from a registered FFI Object type
```
**Workaround.** Always inherit from `tvm_ffi.Object` (or a py_class subclass thereof).

## Enum repr changes if you swap to StrEnum

The default `StrEnum.__repr__` is `<X.A: 'a'>`. If your codebase relies on a
specific repr format (e.g. `X.A` for eval-friendly output), preserve it
explicitly:
```python
class _ReprEnum(StrEnum):
    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"
```

## Dtype validation must allow StrEnum instances

Pre-existing validation that does `isinstance(val, str)` to reject raw
strings will now match StrEnum instances. Add `not isinstance(val, Enum)`
to the guard:
```python
if isinstance(val, str) and not isinstance(val, Enum) and val != _UNSET:
    raise TypeError(...)
```

## Code that calls `.value` or `.c_type` on a former-enum field will break

When you migrate a field type from `DType` (enum) to `str` (StrEnum value),
downstream code that did `expr.dtype.value` or `expr.dtype.c_type` now
operates on a plain string. Look up the enum class explicitly:
```python
# Before:  c_type = expr.dtype.c_type
# After:   c_type = DType(expr.dtype).c_type   # round-trip str → enum → c_type
```

---

## Lessons from the tilus hidet-IR migration

The migration of `python/tilus/hidet/ir/` (a Hidet-style low-level IR
layered on top of tvm-ffi `@py_class`) exposed a cluster of issues that
don't appear in simpler migrations. All of them trace back to the same
underlying shift: an IR node's **Python wrapper is cheap, ephemeral, and
not what identity is really about — the C-object handle is.** Code that
casually used `a is b`, `id(obj)`, or plain Python dict/set on IR values
has to be audited against that shift.

## `is` checks on `node.field` never hold after the migration

Every attribute access on a `@py_class` instance returns a *fresh Python
wrapper* over the same underlying C handle. That means code like
```python
if a is e.a and b is e.b:
    return e
```
never takes the short-circuit path: `e.a` returns a different Python
object on each read, so `a is e.a` is always False, and the rewriter
always rebuilds the expression. In hidet-style compilers this trips two
systems at once:

- **`IRRewriter.visit_*`** (`expr_functor.py`, `stmt_functor.py`): the
  entire "no-op rebuild" short-circuit pattern breaks. Every visit
  produces a new node handle even when none of the children changed.
- **Fixed-point loops** (`repeat_until_converge` in `utils/py.py`, the
  `while True` inside `_rule_based_simplifier_base.visit`): the loop
  condition `orig is cur` never fires, so the simplifier runs forever.
  We saw this as softmax compilation hanging inside pattern matching.

**Workaround.** Introduce a `same_node(a, b)` helper that does
`a is b or (hasattr(a, "same_as") and a.same_as(b))`, and use it (or an
equivalent `.same_as()` branch) everywhere `is` was comparing IR nodes.
Extend `same_list` the same way. Fixed-point loops must check
`same_as` alongside `is`.

## Identity ≠ `id(obj)` for FFI wrappers

`id(obj)` returns the *Python wrapper's address*, not the C-object
pointer. Since wrappers are created fresh per access and destroyed
quickly, CPython reuses their `id`s almost immediately. Any data
structure that keyed on `id(node)` to track "have I seen this node
before" is subtly broken:

- A just-freed wrapper's `id` gets recycled for the *next* wrapper, which
  may correspond to a completely different C handle.
- Dict keyed on `id` happily claims a cache hit on the new wrapper, and
  returns stale data (or, worse, a cached result computed for a
  different object).

This is *exactly* how a functor memo keyed on `id(ffi.Dict)` emptied
`IRModule.functions` at the first pass: `module.global_vars` was visited
first (empty dict, memoized), then `module.functions` was accessed; the
second wrapper happened to inherit the freed `global_vars` wrapper's
`id`; the memo claimed a hit and returned the empty dict.

**Workaround — and the cleanest one possible.** `tvm_ffi.Object`'s
default `__hash__` and `__eq__` are already handle-address-based
(straight from `chandle` in the Cython layer: `hash(obj)` returns the
C pointer, `a == b` returns `a.chandle == b.chandle`). So a plain
Python `dict` keyed on the IR-node itself gives handle-identity
lookup for free, AND storing the node as the key keeps a strong
reference that pins the C handle for the dict's lifetime. No custom
NodeDict / NodeSet wrapper is needed; no `id()`-keyed bookkeeping is
needed; no `__chandle__()` indirection is needed. Reach for
`StructuralKey` only when you explicitly want structural semantics
(CSE, dedup). The only things that still need `id()` keying are plain
Python `list` / `dict` (not hashable at all) and FFI containers
(their `__hash__` is uncached `RecursiveHash`, separate trap below).

## `tvm_ffi.StructuralKey` is a drop-in "structural memo key"

`StructuralKey(obj)` wraps an Object and exposes structural
`__hash__` / `__eq__` based on `structural_hash` / `structural_equal`.
Two important properties:

- It stores a `key: Any` FFI field containing the wrapped Object, so the
  C handle stays alive for the `StructuralKey`'s lifetime.
- Its `hash_i64` is computed once and cached in the FFI struct, so
  repeated `hash()` probes are O(1).

For containers keyed *structurally* (e.g. common-subexpression-elimination
maps, CSE dedup), `{StructuralKey(x): v}` is the clean idiom. We chose
not to use it for pass memos — identity is usually what we need there —
but it's the right tool when two structurally-equal IR fragments should
collapse.

**Gotcha.** `StructuralKey(c)` where `c` is a container recurses through
its elements to compute the structural hash, and will raise `TypeError:
Type metadata is not set for type X, so StructuralHash is not supported`
if *any* nested element's type lacks a `structural_eq="..."`
declaration. Add `structural_eq="tree"` on every py_class in the
container's element graph even when you don't care about structural
equality on that type itself — otherwise `StructuralKey` on an outer
container aborts.

## No custom `NodeDict` / `NodeSet` needed — plain `dict` / `set` work

Because `tvm_ffi.Object`'s default `__hash__` / `__eq__` are
handle-address-based, and Python dicts hold strong references to their
keys (which keep the underlying C handle alive), a plain `dict[Node, V]`
is *already* the identity-keyed map compiler passes want:

- Handle-identity semantics: two wrappers of the same C handle collide
  on lookup; two structurally-equal but distinct nodes stay separate.
- No id-reuse trap: the dict's own reference to the key pins the C
  handle, so its hash (derived from the C pointer) is stable for the
  dict's lifetime.
- No extra indirection: no `NodeDict(...).__setitem__`, no
  `StructuralKey` wrapping, no `(node, result)` value tuples.

We started out with a `NodeDict` / `NodeSet` wrapper (first around
`id(obj)`, then around `StructuralKey`, then around `__chandle__()`)
before realizing that the only workable design — identity on the C
pointer — is what `tvm_ffi.Object` gives you for free. The wrapper
added complexity without buying anything.

Wrappers only pay off when you need *structural* keying or want to
lock down a specific alive-reference invariant at a call site. Both
are rare.

## FFI `Array` / `List` / `Map` / `Dict` are *not* `isinstance` of tuple/list/dict

These classes inherit from `tvm_ffi.Object` + `MutableSequence` /
`MutableMapping`, not from Python's builtin containers. So:

- `isinstance(x, (list, tuple))` is **False** for `ffi.Array`.
- `isinstance(x, dict)` is **False** for `ffi.Dict`.
- They iterate, len-check, and index like Python sequences, but pattern
  matches on the builtins fail.

Every pre-refactor site that did
```python
if isinstance(dims, (list, tuple)):
```
needs an `is_seq(x)` duck-typed helper (returns True for Python
`tuple` / `list` and for any type whose name is `"Array"` / `"List"`).
Otherwise code silently goes down the "scalar" branch with a container
in its hands.

## FFI container `__hash__` is uncached `RecursiveHash`

Unlike `StructuralKey`, plain `ffi.Array` / `List` / `Map` / `Dict`
compute their hash as `_ffi_api.RecursiveHash(self)` on *every* call —
no caching. For an IR rewriter memo that probes "have I seen this
Array-valued field before?", repeated hashing is a real cost.
Additionally, `RecursiveHash` requires every nested element type to
declare a `structural_eq="..."` kind; otherwise it raises.

**Workaround.** Either skip the memo for FFI containers (they're
usually visited once per traversal anyway) or use `__chandle__()`-based
keying with a reference stored on the value side.

## `@py_class`-backed node is immutable — `self.field = ...` raises

Because `frozen=True` (and even when not) the generated `__init__` stores
fields in the C handle, not `__dict__`. Any legacy code that did
`self.gmem_base_ptr = Var(...)` on an IR node (or its subclass) now
raises `AttributeError: ... no attribute '...' and no __dict__`. This
collided unexpectedly with:

- `typing.no_type_check`, which assigns `arg.__no_type_check__ = True`
  on its argument and only suppresses `TypeError` on the assignment. On
  our frozen `Function` instances it raises `AttributeError`.
- Primitive-function registration sites that applied `@no_type_check`
  -after* `@script` — i.e. to the already-compiled hidet `Function`,
  not to the Python function being compiled.

**Workaround.** For `no_type_check`, either reorder decorators (apply
it *before* `@script`) or install a module-level shim that catches the
`AttributeError`:
```python
_real = typing.no_type_check
def _tolerant(arg):
    try: return _real(arg)
    except AttributeError: return arg
typing.no_type_check = _tolerant
```

## Don't assign instance attributes via `self.foo = ...` in user `__init__`

User-defined `__init__` on a `@py_class(init=True)` is preserved (per
the "`__post_init__` IS supported" footgun above), but the body must
call `self.__ffi_init__(*positional_fields)` — it *cannot* do
`self.field = value` for the fields of the py_class, because
frozen/slotted layout rejects attribute sets.

For fields that need pre-normalization (e.g. `convert()` coercion,
string-to-DataType coercion), the pattern is:
```python
@py_class("...", frozen=True, structural_eq="tree")
class DeclareStmt(Stmt):
    var: Var
    init: Optional[Expr] = None
    ...

    def __init__(self, var, init=None, is_static=False, scope=None):
        assert isinstance(var, Var)
        self.__ffi_init__(var, convert(init), is_static, _scope_to_str(scope))
```

## Sequence-flatten guards must accept `ffi.Array` for nested fields

Methods like `LetStmt.__init__(self, bind_vars, ...)` historically
accepted either a scalar or a sequence:
```python
if not isinstance(bind_vars, (list, tuple)):
    bind_vars = (bind_vars,)
```
After migration, `stmt.bind_vars` is an `ffi.Array`, so
`isinstance(bind_vars, (list, tuple))` is False and the single-Array
input gets re-wrapped as `(Array,)` — which then fails at the FFI
level with "expected Var, got Array". Always use the `is_seq()` helper
for this branch.

## Module-level stdlib collisions from expanded FFI import chains

Migrating `tilus.hidet.ir` to import `tvm_ffi.dataclasses` pulled in
tvm-ffi's Cython `init` chain, which (indirectly, via torch) triggers
`import tempfile` → `from random import Random`. A tilus module named
`tilus/hidet/ir/primitives/cuda/random.py` shadowed the stdlib
`random` in the partial-import state and deadlocked with a circular
import. Rename project modules that shadow stdlib names (`random.py` →
`philox.py` in our case) if the FFI import chain reaches them.

## Don't use `isinstance(x, tuple)` for "is this a container of IR nodes"

Use `is_seq(x)` or duck-type on iter/len. The number of places that
need this change is large (analyzers, normalize helpers, builders);
grep each carefully — the symptom is usually a cryptic downstream
"expected Foo, got Bar" at the FFI type-convert boundary, several
frames removed from the actual guard.

## `dataclasses.fields(obj)` fails on py_class

Places that introspected field names via `dataclasses.fields(func.attrs)`
need to switch to an explicit iteration over a `ClassVar` list (or to
`tvm_ffi.dataclasses.fields` if that's a public API). We ended up
adding `_FIELDS: ClassVar[tuple[str, ...]] = ("grid_dim", ...)` on
`FuncAttrs` and iterating that in the printer.
