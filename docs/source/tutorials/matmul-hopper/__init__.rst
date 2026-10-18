Matmul (Hopper)
===============

This tutorial shows how to implement a high-performance matrix multiplication kernel
(C = A x B\ :sup:`T`) targeting **NVIDIA Hopper GPUs** using **Tilus**.

Starting from a minimal working kernel, each version introduces one new Hopper feature
or optimization technique. By the final version, the kernel exceeds vendor-library
performance for this shape. The figure below shows the progression: V0 starts at
~305 TFLOPS with a minimal kernel that pushes every operand through the register file,
and each optimization closes the gap to cuBLAS, with V6 passing it at ~756 TFLOPS.
All kernels and the benchmark script to reproduce the result can be found at
:github:`examples/hopper_matmul/`.

.. note::

   Every version, like cuBLAS, takes fp16 inputs, accumulates in fp32, and writes
   fp16 outputs, so all comparisons are like-for-like. :doc:`V6 <v6>` notes what
   switching to fp16 accumulation would buy, and what it would cost.

.. plot:: tutorials/matmul-hopper/plots/plot_all.py

   Hopper matmul performance on H100 SXM (M=N=K=8192, fp16). Latency is
   CUDA-event timed, median of three fresh processes. Peak is the published
   dense FP16 tensor core throughput of the H100 SXM.

The progression is not perfectly monotonic: V2 introduces multi-stage pipelining but
measures slightly slower than V1, because a ring buffer alone does not create overlap
when all threads still meet at a block-wide barrier. :doc:`V3 <v3>` supplies the
missing half and both changes pay off together. The :doc:`V2 <v2>` page works through
this in detail --- it is the most instructive step in the series.

.. list-table:: Summary (H100 SXM, M=N=K=8192, fp16)
   :header-rows: 1
   :widths: 8 30 10 14 12 12

   * - Version
     - Optimization
     - Accum.
     - Latency
     - TFLOPS
     - Tensor pipe
   * - :doc:`V0 <v0>`
     - TMA loads, register-staged ``mma.sync``
     - fp32
     - 3.60 ms
     - 305
     - 59%
   * - :doc:`V1 <v1>`
     - WGMMA from shared memory
     - fp32
     - 2.04 ms
     - 540
     - 67%
   * - :doc:`V2 <v2>`
     - Multi-stage software pipelining
     - fp32
     - 2.12 ms
     - 518
     - 71%
   * - :doc:`V3 <v3>`
     - Warp specialization
     - fp32
     - 1.91 ms
     - 575
     - 75%
   * - :doc:`V4 <v4>`
     - Two consumers, ``Pipeline`` class, tile rasterization
     - fp32
     - 1.71 ms
     - 642
     - 80%
   * - :doc:`V5 <v5>`
     - Overlapped WGMMA groups
     - fp32
     - 1.62 ms
     - 678
     - 88%
   * - :doc:`V6 <v6>`
     - Register reallocation, 2-CTA multicast, persistent tiles
     - fp32
     - **1.45 ms**
     - **756**
     - 94%
   * - cuBLAS
     - ``nvjet_sm90_hsh_320x128_64x3_1x2_h_bz_coopB_TNT``
     - fp32
     - 1.46 ms
     - 752
     - 93%

Tensor pipe utilization is from Nsight Compute
(``sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed``); the latency and
TFLOPS columns are CUDA-event timings, median of three fresh processes. Reproduce
with::

   python examples/hopper_matmul/benchmark.py --size 8192 8192 8192
   python examples/hopper_matmul/benchmark.py --ncu --size 8192 8192 8192

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
