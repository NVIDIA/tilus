Tensor Memory Tensor
====================

A tensor memory tensor (i.e., :py:class:`~tilus.ir.tensor.TMemoryTensor`) is a tensor stored in the **Tensor Memory (TMEM)**
of Blackwell GPUs (sm_100+). Tensor Memory is a dedicated on-chip memory specialized for use by the 5th-generation
Tensor Cores (``tcgen05``).

.. figure:: /python-api/instruction-groups/figures/tmem_layout.svg
   :width: 80%
   :align: center

   Tensor Memory layout: 128 lanes x 512 columns, each cell 32 bits.

- **dtype**: the data type of the tensor elements.
- **shape**: the shape of the tensor. The second-to-last dimension (``shape[-2]``) must be 32, 64, or 128.

Tensor Memory is organized as a 2D structure of **128 rows** (called *lanes*) and **512 columns** per CTA,
with each cell being 32 bits. Memory is allocated in units of 32 columns.



Tensor Memory Instructions
--------------------------

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

All tensor memory allocated in a kernel must be explicitly deallocated before the kernel exits.

For more details, see :doc:`/python-api/instruction-groups/tcgen05`.
