Cache
=====

Tilus caches the results of compilation to speed up subsequent calls to the same tilus script. By default, tilus stores
the cache under the **default cache directory**: ``~/.cache/tilus``.

You can change the cache directory by calling the :py:func:`tilus.option.cache_dir`
function.

Get the generated CUDA kernel
-----------------------------

A convenient way to get the generated CUDA kernel is to change the cache directory of tilus to the current working directory:

.. code-block:: python

    import tilus
    tilus.option.cache_dir('./cache')

    <your program>

This will store the generated CUDA kernel in the ``./cache`` directory, and you can find the generated cuda kernels.

If you are interested in the compilation process, you can also enable the debug mode to dump the
intermediate representations (IRs) via the :py:func:`tilus.option.debug.dump_ir` function.

.. code-block:: python

    import tilus

    tilus.option.cache_dir('./cache')
    tilus.option.debug.dump_ir(True)

    <your program>

Cache Structure
---------------

The cache directory contains the following structure (only show the important parts):

- **cache/**

  - **scripts/**: compilation results for all tilus scripts

    - **<script-name>/**: name of your tilus script class, converted to snake case

      - **<script-hash>/**

        - **programs/**: programs for each schedule

          - **0** a soft link to a program directory for the **first** schedule
          - **1** a soft link to a program directory for the **second** schedule
          - ...

        - **dispatch_table.txt** dispatch table from dynamic input size to program id

  - **programs/** compilation results for all tilus programs, each corresponds to a schedule of a tilus script

    - **<program-hash>/**  hex256 hash of the program.txt and options.txt in the directory

      - **ir/** IRs, when dump_ir is enabled
      - **module/**

        - **ir/**  IRs, when dump_ir is enabled
        - **compile.sh**  the bash script to compile the source.cu into lib.so
        - **source.cu**   the source code of the compiled CUDA kernel
        - **lib.so**  the shared library with the compiled CUDA kernel

      - **program.txt** the program text, which is a human-readable representation of the tilus program
      - **options.txt** the options used to compile the program

    - ...
