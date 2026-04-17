Matmul (Blackwell)
==================

This tutorial shows how to implement a high-performance matrix multiplication kernel
(C = A x B\ :sup:`T`) targeting **NVIDIA Blackwell GPUs** using **Tilus**.

Starting from a minimal working kernel, each version introduces one new Blackwell feature
or optimization technique. By the final version, the kernel reaches vendor-library-level
performance. The figure below shows the progression: V0 starts at ~491 TFLOPS with a
minimal kernel, and each optimization closes the gap to cuBLAS, with V6 matching it at
~1610 TFLOPS. All kernels and the benchmark script to reproduce the result can be found at :github:`examples/blackwell_matmul/`.

.. plot:: tutorials/matmul-blackwell/plots/plot_all.py

   Blackwell matmul performance on B200 (M=N=K=8192, fp16). TFLOPS derived
   from NCU profiling. Peak TFLOPS estimated from cuBLAS tensor core
   utilization (96.6%).

.. toctree::
   :maxdepth: 1
   :caption: Versions

   v0
   v1
   v2
   v3
   v4
   v5
   v6
