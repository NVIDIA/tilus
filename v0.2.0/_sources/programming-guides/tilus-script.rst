Tilus Script
============

Tilus Script is a flexible scripting language designed for simplifying the GPU development without sacrificing
performance. It is a domain-specific language (DSL) embedded in Python, allowing developers to write GPU kernels in a
more intuitive and Pythonic way.

To define a kernel in Tilus Script, you create a subclass of :py:class:`tilus.Script` and implement the ``__init__`` and
``__call__`` method.

.. code-block:: python

    import tilus

    class MyKernel(tilus.Script):
        def __init__(self, *args, **kwargs):
            super().__init__()
            ... # perform any compilation-time setup here

        def __call__(self, param0: type0, param1: type1, ...):
            ...  # kernel code goes here

An example of a simple kernel that adds one to each element of an input tensor is shown below.
This example demonstrates the basic structure of a Tilus script, including defining the kernel and launching it.

.. testcode:: python

    import torch
    import tilus
    from tilus import float32, int32
    from tilus.utils import cdiv


    class AddOneKernel(tilus.Script):
        def __init__(self, block_n, warps):
            super().__init__()
            self.block_n: int = block_n
            self.warps: int = warps

        def __call__(self, n: int32, a_ptr: ~float32, b_ptr: ~float32):
            self.attrs.blocks = cdiv(n, self.block_n)    # define the number of thread blocks
            self.attrs.warps = self.warps                # define the number of warps per block

            # get the offset for the current block
            offset = self.blockIdx.x * self.block_n

            # create two global tensors for input and output, given their pointers
            ga = self.global_view(a_ptr, shape=[n], dtype=float32)
            gb = self.global_view(b_ptr, shape=[n], dtype=float32)

            # load the inputs from global memory into a register tensor
            a = self.load_global(ga, offsets=[offset], shape=[self.block_n])

            # perform the computation: add 1 to each element in the register tensor
            b = a + 1.0

            # store the result back to global memory
            self.store_global(gb, b, offsets=[offset])


    def main():
        # define the kernel
        kernel = AddOneKernel(block_n=128, warps=4)

        # create input and output tensors
        n = 16
        a = torch.arange(n, dtype=torch.float32)
        b = torch.empty_like(a)

        # launch the kernel
        kernel(n, a, b)

        print(a)
        print(b)

    main()


We can run the above code and get the following output:

.. testoutput:: python

    tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
            14., 15.])
    tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.,
            15., 16.])

Script methods
--------------

``__init__`` method
~~~~~~~~~~~~~~~~~~~

When we instantiate a tilus script, its ``__init__`` method is called to perform any compilation-time setup.
This is where you can perform pre-computation over the hyper-parameters of the kernel and use the results in the
``__call__`` method, or simply record the hyper-parameters.


``__call__`` method
~~~~~~~~~~~~~~~~~~~

The ``__call__`` method defines the actual kernel code that will be executed on the GPU. For each kernel, we must
specify the following attributes:

- ``self.attrs.blocks``: the number of thread blocks to launch. It can be a 1, 2, or 3 numbers representing the number
  of blocks in each dimension (x, y, z). When less than 3 numbers are provided, we use 1 for the other dimensions. We
  can use any expression around the kernel parameters to compute the number of blocks, such as ``cdiv(n, self.block_n)``.
- ``self.attrs.warps``: the number of warps per thread block. It must be a compilation-time positive integer constant.
  On NVIDIA GPUs, a warp is a group of 32 threads and the number of warps per block must be in [1, 32], inclusive.


Just-in-time compilation
------------------------

Tilus scripts are compiled just-in-time (JIT) when they are called for the first time. JIT compilation allows for
generating kernels with specific dimensions known at compilation time, which can lead to better performance.
Tilus scripts require all kernel parameters to be annotated with types. To mark some parameters as compilation-time
constants, we can use

- **JIT Annotations**: ``int``, ``float``, or ``bool``
- **Non-JIT Annotations**: ``int32``, ``float32``, ``boolean``, etc. See :doc:`type-system/__init__` for supported types.

The following example demonstrates how to write a matrix multiplication kernel using Tilus Script, with ``m_size`` as
dynamic size, and ``n_size`` and ``k_size`` as compilation-time constants.
When a tilus script is called with different
combination of (``n_size``, ``k_size``) pairs, jit-in-time compilation will be triggered to generate a new kernel for
each unique combination of (``n_size``, ``k_size``). Knowing the ``n_size`` and ``k_size`` at compilation time allows
tilus compiler to optimize the kernel based on their divisibility and enabling vectorized memory loading.
When it's called with different ``m_size``, the same kernel will be reused and no JIT compilation will be triggered.

.. testcode:: python

    import math
    import torch
    import tilus
    from tilus import float16, float32, int32
    from tilus.utils import cdiv


    class Matmul(tilus.Script):
        def __init__(self):
            super().__init__()
            self.block_m = 64
            self.block_n = 128
            self.block_k = 16

        def __call__(self,
                m_size: int32, n_size: int, k_size: int,
                a_ptr: ~float16, b_ptr: ~float16, c_ptr: ~float16
        ):
            self.attrs.blocks = [
                cdiv(m_size, self.block_m),  # the x dimension size of the grid
                cdiv(n_size, self.block_n),  # the y dimension size of the grid
            ]
            self.attrs.warps = 4

            offset_m: int32 = self.block_m * self.blockIdx.x
            offset_n: int32 = self.block_n * self.blockIdx.y

            # create two global tensors `ga` and `gb`
            ga = self.global_view(a_ptr, dtype=float16, shape=[m_size, k_size])
            gb = self.global_view(b_ptr, dtype=float16, shape=[k_size, n_size])

            # create a register tensor `acc` for accumulating the results.
            acc = self.register_tensor(
                dtype=float32, shape=[self.block_m, self.block_n], init=0.0
            )

            # iterate over the k dimension in blocks of size `block_k`.
            for k in range(cdiv(k_size, self.block_k)):
                # calculate the offset for the current block in the k dimension
                offset_k = k * self.block_k

                # load a block of matrix A and B into register tensors `a` and `b`.
                a = self.load_global(
                    ga, offsets=[offset_m, offset_k], shape=[self.block_m, self.block_k]
                )
                b = self.load_global(
                    gb, offsets=[offset_k, offset_n], shape=[self.block_k, self.block_n]
                )

                # perform the dot product: acc = a @ b + acc
                self.dot(a, b, acc, out=acc)

            # after the loop, we cast the accumulated result `acc` to float16 type
            acc_f16 = self.cast(acc, dtype=float16)

            # store it back to the output matrix C.
            gc = self.global_view(c_ptr, dtype=float16, shape=[m_size, n_size])
            self.store_global(gc, acc_f16, offsets=[offset_m, offset_n])


    def main():
        kernel = Matmul()

        for k_size, n_size in [(4096, 4096), (4096, 12288)]:
            for m_size in [1, 4, 8, 16]:
                a = torch.randn(m_size, k_size, dtype=torch.float16, device='cuda') / math.sqrt(k_size)
                b = torch.randn(k_size, n_size, dtype=torch.float16, device='cuda') / math.sqrt(k_size)
                c = torch.empty(m_size, n_size, dtype=torch.float16, device='cuda')

                kernel(m_size, n_size, k_size, a, b, c)
                torch.testing.assert_close(c, torch.matmul(a, b), rtol=1e-2, atol=1e-2)

    main()
