2. Auto-tuning
==============

In previous versions of the matmul kernel, we manually set the hyperparameters
such as ``block_m``, ``block_n``, and ``block_k``. However, these
hyperparameters can significantly affect the performance of the kernel, and
finding the optimal values for them can be a tedious and time-consuming process.

Tilus provides the :meth:`tilus.autotune` decorator to annotate the search space
of the hyperparameters and let tilus automatically search for the best
configuration.

The decorator accepts parameter names and a list of values. When multiple
``@tilus.autotune`` decorators are stacked, tilus forms the Cartesian product of
all value lists and tries every combination. At the first invocation the kernel
is compiled for each configuration, benchmarked on the actual arguments, and the
fastest configuration is selected automatically. Subsequent calls reuse the
winner.

.. code-block:: python

   @tilus.autotune("arg_name1", [v11, v12, v13])
   @tilus.autotune("arg_name2, arg_name3", [(v21, v31), (v22, v32)])
   class AwesomeKernel(tilus.Script):
       def __init__(self, user_arg, arg_name1, arg_name2, arg_name3):
           super().__init__()
           ...

When instantiating the class, only the non-tuned arguments are provided --
the tuned parameters are filled in automatically by the autotuning engine.

Imports
~~~~~~~

.. literalinclude:: ../../../../../examples/matmul/matmul_v2.py
   :language: python
   :lines: 43-45

Annotate the Search Space
~~~~~~~~~~~~~~~~~~~~~~~~~

Reusing the same kernel implementation as in V1, we add
:meth:`tilus.autotune` decorators for ``num_warps``, ``block_m``/``block_n``,
and ``block_k``:

.. literalinclude:: ../../../../../examples/matmul/matmul_v2.py
   :language: python
   :lines: 55-123
   :emphasize-lines: 1-3

The three decorators create a space of :math:`2 \times 3 \times 2 = 12`
configurations. Tilus compiles all twelve, benchmarks them, and keeps the
fastest.

Launch the Kernel
~~~~~~~~~~~~~~~~~

Notice that ``MatmulV2()`` is instantiated with **no arguments** -- the
tuned parameters are determined automatically.

.. literalinclude:: ../../../../../examples/matmul/matmul_v2.py
   :language: python
   :lines: 137-168
   :emphasize-lines: 9

The first call to ``matmul(m, n, k, a, b, c_actual)`` triggers the
autotuning process: every configuration is compiled and benchmarked on the
given arguments. The best configuration is then cached, so subsequent
invocations skip tuning entirely.

.. note::

   The full source code for this example can be found at
   :download:`matmul_v2.py <../../../../../examples/matmul/matmul_v2.py>`.
