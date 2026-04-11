Matrix multiplication
=====================

This tutorial shows how to implement matrix multiplication kernel (C = A x B) in
**tilus**.
We start with a naive kernel and, by adding one optimization per version, reach *cuBLAS* speeds on modern GPUs.

.. toctree::
   :maxdepth: 1
   :caption: Versions

   matmul/matmul_v0
   matmul/matmul_v1
   matmul/matmul_v2
   matmul/matmul_v3
   matmul/matmul_v4
   matmul/matmul_v5
