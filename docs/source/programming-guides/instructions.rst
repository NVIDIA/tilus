Instructions
============

Tilus provides a set of instructions for writing GPU kernels. Instructions are available as methods
on the :class:`~tilus.Script` class and are called within the ``__call__`` method of a script.

Instructions fall into two categories:

- **Generic instructions** (``self.<instruction>``) — common operations available on all GPUs, such as
  tensor creation, load/store, arithmetic, and synchronization.
- **Instruction groups** (``self.<group>.<instruction>``) — specialized hardware-specific operations
  organized by the hardware unit they target, such as TMA, WGMMA, and TCGEN05.


Generic Instructions
--------------------

.. hint::
   :class: margin

   Please submit a feature request if your kernel requires additional instructions.

.. currentmodule:: tilus.Script


Tensor Creation and Free
~~~~~~~~~~~~~~~~~~~~~~~~~

Create and manage tensors in register, shared, and global memory. Register tensors hold per-thread
data, shared tensors are visible to all threads in a block, and global tensors are accessible by all
blocks.

.. autosummary::

   register_tensor
   shared_tensor
   global_tensor
   global_view
   free_shared
   reshape_shared


Load and Store
~~~~~~~~~~~~~~

Transfer data between memory spaces. Load instructions copy data from global or shared memory into
register tensors; store instructions write register data back. The ``*_scatter`` variants perform
non-atomic scatter writes driven by a per-lane ``indices`` tile along one axis — use them when the
scatter is guaranteed collision-free, and reach for :ref:`self.atomic.*_scatter_* <atomic-group>`
otherwise.

.. autosummary::

   load_global
   store_global
   store_global_scatter
   load_shared
   store_shared
   store_shared_scatter


Asynchronous Copy (SM80+)
~~~~~~~~~~~~~~~~~~~~~~~~~

Copy data from global to shared memory asynchronously using the ``cp.async`` hardware path.
Operations are grouped with ``copy_async_commit_group`` and waited on with ``copy_async_wait_group``.
For Hopper+ GPUs, prefer ``tma.global_to_shared`` which uses the TMA engine.

.. autosummary::

   copy_async
   copy_async_commit_group
   copy_async_wait_group
   copy_async_wait_all


Linear Algebra
~~~~~~~~~~~~~~

Matrix multiplication using tensor cores. The ``dot`` instruction automatically selects the
appropriate MMA instruction based on the data types and GPU architecture. For explicit control
over Hopper or Blackwell tensor cores, use ``wgmma.mma`` or ``tcgen05.mma`` instead.

.. autosummary::

   dot


Elementwise Arithmetic
~~~~~~~~~~~~~~~~~~~~~~

Per-element unary and binary operations on register tensors. All elementwise operations support
an optional ``out`` parameter to write results into an existing tensor, and binary operations
support NumPy-style broadcasting.

.. autosummary::

   abs
   add
   clip
   cos
   exp
   exp2
   log
   maximum
   round
   rsqrt
   sin
   sqrt
   square
   where


Random Number Generation
~~~~~~~~~~~~~~~~~~~~~~~~

Generate pseudo-random numbers using the Philox-4x32-10 counter-based PRNG. Given a seed (uint64
scalar) and an offset tensor (uint32), each element produces independent random values.
``randint4x`` is the most efficient entry point, returning all four Philox outputs per invocation.
``rand`` and ``randn`` build on top to provide uniform and normal distributions respectively.

.. autosummary::

   rand
   randint
   randint4x
   randn


Reduction
~~~~~~~~~

Reduce a register tensor along one or more dimensions. Each reduction supports ``dim`` to specify
which dimensions to reduce, ``keepdim`` to preserve the reduced dimension with size 1, and ``out``
for in-place output.

.. autosummary::

   all
   any
   max
   min
   sum


Prefix Scan
~~~~~~~~~~~

Tile-level prefix scan along one dimension. The output preserves the input's shape; along ``dim``,
element ``i`` holds the ⊕-combination of positions ``[0, i]`` (inclusive) or ``[0, i)`` with the
op's identity at ``i == 0`` (exclusive). Supported ops: ``add`` / ``mul`` / ``max`` / ``min`` plus
the bitwise ``and`` / ``or`` / ``xor`` for integer tiles. The lowering is Blelloch's up-sweep +
down-sweep, bit-local at every round, so every valid tilus register layout is handled — intra-
thread, intra-warp, inter-warp, and any mixture. ``cumsum`` / ``cumprod`` are convenience
shortcuts for ``op='add'`` / ``op='mul'``.

.. autosummary::

   scan
   cumsum
   cumprod


Transform
~~~~~~~~~

Reshape, reinterpret, or rearrange register tensor data without changing the underlying values.

.. autosummary::

   assign
   cast
   repeat
   repeat_interleave
   squeeze
   transpose
   unsqueeze
   view


Synchronization
~~~~~~~~~~~~~~~

Synchronize threads within a block or across a cluster. ``sync`` is the block-level barrier
(equivalent to ``__syncthreads()``). For cluster-wide synchronization, use ``self.cluster.sync()``.

.. autosummary::

   sync


Semaphores
~~~~~~~~~~

Inter-block synchronization using global memory semaphores. ``lock_semaphore`` spins until the
semaphore reaches a target value; ``release_semaphore`` sets it to signal other blocks. Both
must be called from a single thread (``self.single_thread()``). For tile-level atomic RMW (element-wise
and scatter), see :ref:`self.atomic <atomic-group>`.

.. autosummary::

   lock_semaphore
   release_semaphore


Miscellaneous
~~~~~~~~~~~~~

Compiler hints, debugging aids, and layout annotations.

.. autosummary::

   assume
   static_assert
   annotate_layout
   fast_divmod
   print_tensor
   printf


Instruction Groups
------------------

Instruction groups provide access to specialized hardware units. Each group is accessed as an
attribute of the script (e.g., ``self.tma.global_to_shared(...)``).


Memory Barrier (``self.mbarrier``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mbarriers are synchronization primitives in shared memory that track pending arrivals and
asynchronous transaction bytes (tx-count). They coordinate producer-consumer patterns in pipelined
kernels, particularly with TMA and TCGEN05 async operations. See :doc:`../python-api/instruction-groups/mbarrier`.

.. currentmodule:: tilus.lang.instructions.mbarrier.BarrierInstructionGroup

.. autosummary::

   alloc
   arrive
   arrive_and_expect_tx
   arrive_and_expect_tx_multicast
   arrive_and_expect_tx_remote
   wait

.. currentmodule:: tilus.Script


Fence (``self.fence``)
~~~~~~~~~~~~~~~~~~~~~~

Proxy fences ensure memory ordering between different memory access paths (generic proxy vs.
async proxy). Required when generic writes (e.g., ``store_shared``) must be visible to async
reads (e.g., ``tma.shared_to_global``). See :doc:`../python-api/instruction-groups/fence`.

.. currentmodule:: tilus.lang.instructions.fence.FenceInstructionGroup

.. autosummary::

   proxy_async
   proxy_async_release

.. currentmodule:: tilus.Script


TMA (``self.tma``)
~~~~~~~~~~~~~~~~~~

The Tensor Memory Accelerator (TMA) on Hopper+ GPUs performs asynchronous bulk data transfers
between global and shared memory without occupying SM compute resources. Completion is tracked
via mbarriers. See :doc:`../python-api/instruction-groups/tma`.

.. currentmodule:: tilus.lang.instructions.tma.TmaInstructionGroup

.. autosummary::

   global_to_shared
   shared_to_global
   commit_group
   wait_group

.. currentmodule:: tilus.Script


WGMMA (``self.wgmma``)
~~~~~~~~~~~~~~~~~~~~~~~

Warp Group Matrix Multiply-Accumulate on Hopper GPUs. Executes asynchronous MMA using a warp group
(4 warps, 128 threads) with operands in shared memory or registers. Requires a strict
fence → mma → commit → wait protocol. See :doc:`../python-api/instruction-groups/wgmma`.

.. currentmodule:: tilus.lang.instructions.wgmma.WgmmaInstructionGroup

.. autosummary::

   fence
   commit_group
   wait_group
   mma

.. currentmodule:: tilus.Script


TCGEN05 (``self.tcgen05``)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor Core Generation 05 on Blackwell GPUs. Introduces tensor memory (TMEM) — a dedicated
on-chip accumulator space for MMA operations. Supports the full TMEM lifecycle: allocation,
data movement (load/store/copy), MMA compute, and deallocation. See :doc:`../python-api/instruction-groups/tcgen05`.

.. currentmodule:: tilus.lang.instructions.tcgen05.Tcgen05InstructionGroup

.. autosummary::

   alloc
   dealloc
   slice
   view
   relinquish_alloc_permit
   load
   store
   wait_load
   wait_store
   copy
   commit
   mma

.. currentmodule:: tilus.Script


Cluster (``self.cluster``)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Block cluster operations for multi-CTA coordination on Hopper+ GPUs. Provides cluster-wide
synchronization, introspection (block index/rank within the cluster), and cross-CTA shared memory
addressing. See :doc:`../python-api/instruction-groups/cluster`.

.. currentmodule:: tilus.lang.instructions.cluster.BlockClusterInstructionGroup

.. autosummary::

   sync
   map_shared_addr
   blockIdx
   blockRank
   clusterDim

.. currentmodule:: tilus.Script


CLC (``self.clc``)
~~~~~~~~~~~~~~~~~~

Cluster Launch Control on Blackwell GPUs enables dynamic work scheduling by canceling
not-yet-launched clusters. A scheduler CTA requests cancellation, then queries the result to
take over the canceled cluster's work. See :doc:`../python-api/instruction-groups/clc`.

.. currentmodule:: tilus.lang.instructions.clc.ClusterLaunchControlInstructionGroup

.. autosummary::

   try_cancel
   query_response

.. currentmodule:: tilus.Script


.. _atomic-group:

Atomic (``self.atomic``)
~~~~~~~~~~~~~~~~~~~~~~~~

Tile-level atomic read-modify-write on shared and global memory. Two shapes are exposed:
**element-wise** (``dst.shape == values.shape``, each lane atomically updates its own element)
and **scatter** (``torch.scatter_add_``-style, with a per-lane ``indices`` tile picking the
destination along a compile-time ``dim``). Each method takes ``sem`` / ``scope`` PTX qualifiers and
an optional ``output`` register that receives the pre-RMW value — when unused, codegen lowers to
the cheaper destination-less ``red.*`` form automatically. See
:doc:`../python-api/instruction-groups/atomic`.

.. currentmodule:: tilus.lang.instructions.atomic.AtomicInstructionGroup

.. autosummary::

   shared_add
   shared_sub
   shared_min
   shared_max
   shared_exch
   shared_cas
   global_add
   global_sub
   global_min
   global_max
   global_exch
   global_cas
   shared_scatter_add
   shared_scatter_sub
   shared_scatter_min
   shared_scatter_max
   global_scatter_add
   global_scatter_sub
   global_scatter_min
   global_scatter_max

.. currentmodule:: tilus.Script
