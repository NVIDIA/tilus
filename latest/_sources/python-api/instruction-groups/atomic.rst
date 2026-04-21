Script.atomic
=============

.. currentmodule:: tilus.lang.instructions.atomic

.. autoclass:: AtomicInstructionGroup
   :no-autosummary:
   :no-members:
   :exclude-members: __init__, __new__


Element-wise vs. scatter
------------------------

Two shapes of tile-level atomic RMW are exposed:

**Element-wise** --- :meth:`~AtomicInstructionGroup.shared_add` and friends.
``dst.shape == values.shape`` and each lane contributes its own slice of
``values`` into the matching slice of ``dst`` under the chosen PTX
``atom.*`` op. There is no broadcast and no reduction: this is the right
primitive when each address is written independently, e.g. updating a
per-element counter or applying a reduction across thread groups.

.. code-block:: python

   # Each lane contributes its own value to dst[lane]; one atomic op per lane.
   self.atomic.shared_add(dst, values)

**Scatter** --- :meth:`~AtomicInstructionGroup.shared_scatter_add` and
friends. Modelled after ``torch.scatter_add_``: a compile-time ``dim`` plus
an ``indices`` register tile picks the destination along that axis, while
the non-scatter axes come from the lane's own tile position. This is the
right primitive for histograms and other data-dependent address patterns.

.. code-block:: python

   # Each lane picks bin = indices[lane] and atomic-adds 1 into that bin.
   self.atomic.shared_scatter_add(
       hist, dim=0, indices=bins, values=ones)


Op family
---------

The element-wise family is ``add`` / ``sub`` / ``min`` / ``max`` / ``exch``
/ ``cas``. The scatter family drops ``exch`` and ``cas`` --- their
semantics under duplicate indices are not well defined --- and keeps
``add`` / ``sub`` / ``min`` / ``max``.

PTX has no native ``atom.sub``, so ``sub`` variants lower to ``atom.add``
with a negated operand at codegen time.

In v1 only the ``int32`` dtype is supported. ``float32`` and ``uint32``
coverage can be added by extending the dtype table in the underlying hidet
primitive layer.


sem and scope qualifiers
------------------------

All instructions accept two PTX qualifiers:

- ``sem``: memory-ordering qualifier, one of ``'relaxed'``, ``'acquire'``,
  ``'release'``, ``'acq_rel'``. Defaults to ``'relaxed'``. Matches the
  ``atom.{sem}.*`` PTX syntax.
- ``scope``: sync-scope qualifier, one of ``'cta'``, ``'cluster'``,
  ``'gpu'``, ``'sys'``. Defaults to ``'cta'`` on shared-memory ops and
  ``'gpu'`` on global-memory ops.

Both are passed through to the generated ``atom.{sem}.{scope}.{space}.
{op}.{dtype}`` (or ``red.*``) instruction.


Optional output register
------------------------

Every atomic method accepts an optional ``output`` register tile that
receives the **pre-RMW** value at each destination location. When the
returned register is not consumed by any downstream instruction, the DCE
pass rewrites the instruction to carry ``output=None`` and codegen
lowers it to the destination-less ``red.*`` PTX form instead of
``atom.*``. The net effect is that you only pay for the register return
when your code actually uses it:

.. code-block:: python

   # No caller reads the return value → lowers to `red.*`.
   self.atomic.shared_scatter_add(hist, dim=0, indices=bins, values=ones)

   # Caller consumes `old` → lowers to `atom.*` with a destination register.
   old = self.atomic.shared_cas(lock, compare=zero, values=one)
   with self.if_then(old == 0):
       ...

``exch`` and ``cas`` have no ``red.*`` counterpart in PTX, so their
output is effectively always bound --- if unused the register simply goes
to waste.


Related: non-atomic scatter stores
----------------------------------

When the scatter targets are guaranteed collision-free (e.g. a
permutation), the cheaper non-atomic variants are available directly on
``self``:

- :meth:`tilus.Script.store_shared_scatter`
- :meth:`tilus.Script.store_global_scatter`

They share the same ``dim`` / ``indices`` / ``values`` contract as the
atomic scatter ops. Under duplicate ``indices`` they give last-writer-wins
with an unspecified winner; use the atomic form if correctness under
duplicates matters.


Instructions
------------

.. currentmodule:: tilus.lang.instructions.atomic.AtomicInstructionGroup

.. rubric:: Element-wise (shared memory)

.. autosummary::
   :toctree: generated

   shared_add
   shared_sub
   shared_min
   shared_max
   shared_exch
   shared_cas

.. rubric:: Element-wise (global memory)

.. autosummary::
   :toctree: generated

   global_add
   global_sub
   global_min
   global_max
   global_exch
   global_cas

.. rubric:: Scatter (shared memory)

.. autosummary::
   :toctree: generated

   shared_scatter_add
   shared_scatter_sub
   shared_scatter_min
   shared_scatter_max

.. rubric:: Scatter (global memory)

.. autosummary::
   :toctree: generated

   global_scatter_add
   global_scatter_sub
   global_scatter_min
   global_scatter_max
