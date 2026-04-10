.. _tutorial_matmul_v0:

V0: Naive Matmul
================

This tutorial demonstrates a simple implementation of matrix multiplication
using Tilus. We use this example to illustrate the basic concepts of writing a
kernel in Tilus, including kernel definition, data types, tensors, and kernel
invocation.


Tilus Script
------------

In Tilus, every kernel is defined by subclassing :class:`tilus.Script`. A script
has two methods that you must implement:

- ``__init__`` -- initializes the compilation-time hyperparameters of the
  script (tile sizes, pipeline depths, etc.).
- ``__call__`` -- the main entry point that describes the computation logic of
  the kernel.

The skeleton looks like this:

.. literalinclude:: ../../../../../examples/matmul/matmul_v0.py
   :language: python
   :lines: 24-36
   :caption: Script skeleton


Naive Matmul Implementation
---------------------------

With the script concept in mind, let us implement a naive matrix multiplication
kernel. This implementation is not optimized for performance, but it serves as a
good starting point to understand the basics of Tilus.

We begin with the necessary imports:

.. literalinclude:: ../../../../../examples/matmul/matmul_v0.py
   :language: python
   :lines: 44-46

The full kernel class is shown below. It tiles the output matrix into blocks of
size ``block_m x block_n`` and iterates over the K dimension in chunks of
``block_k``. Each thread block computes one output tile by accumulating partial
dot products in a register tensor.

.. literalinclude:: ../../../../../examples/matmul/matmul_v0.py
   :language: python
   :lines: 48-104
   :caption: MatmulV0 kernel


Type Annotations and Instructions
---------------------------------

.. currentmodule:: tilus

The ``__call__`` signature uses three kinds of type annotations for the kernel
parameters:

- ``int32`` -- a runtime-known 32-bit integer. In the example this is used for
  ``m_size``.
- ``int`` -- a compile-time-known integer. Different values trigger
  Just-In-Time (JIT) re-compilation of the kernel. In the example this is used
  for ``n_size`` and ``k_size``.
- ``~float16`` -- a pointer to a ``float16`` array (equivalent to ``float16*``
  in C/C++). In the example this is used for ``a_ptr``, ``b_ptr``, and
  ``c_ptr``.

.. currentmodule:: tilus.Script

The kernel body uses the following Tilus instructions:

- :meth:`global_view` -- create a global tensor view of the input/output
  matrices.
- :meth:`register_tensor` -- allocate a register tensor to accumulate partial
  results.
- :meth:`load_global` -- load a tile from a global tensor into a register
  tensor.
- :meth:`dot` -- perform a matrix multiply-accumulate on two register tensors.
- :meth:`cast` -- cast a register tensor to a different data type.
- :meth:`store_global` -- store a register tensor back to a global tensor.

All of these instructions have **block semantics**: they are collectively
executed by every thread in the thread block.


Launching the Kernel
--------------------

To launch the kernel, create an instance of ``MatmulV0`` and call it with the
appropriate arguments. The code below also verifies correctness and benchmarks
the kernel against PyTorch:

.. literalinclude:: ../../../../../examples/matmul/matmul_v0.py
   :language: python
   :lines: 135-174
   :caption: Launching, correctness check, and benchmark

The kernel is invoked just like a regular Python function -- Tilus handles grid
configuration and kernel dispatch behind the scenes. The call
``matmul(m, n, k, a, b, c_actual)`` launches the GPU kernel with the specified
parameters, and the results are checked for correctness using
``torch.testing.assert_close``.

The output is a ``pandas.DataFrame`` that contains the latency and throughput
(TFLOPS) of both the Tilus kernel and the PyTorch (cuBLAS) baseline. This naive
kernel is not yet competitive with vendor libraries, but it establishes the
foundation we build on. In subsequent versions we will introduce optimizations
-- shared memory tiling, software pipelining, TMA loads, and more -- that bring
performance up to cuBLAS levels.


Full Source
-----------

The complete example file is located at
:download:`examples/matmul/matmul_v0.py <../../../../../examples/matmul/matmul_v0.py>`.
