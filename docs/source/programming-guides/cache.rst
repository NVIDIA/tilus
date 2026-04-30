Cache
=====

Tilus caches compiled kernels so that subsequent calls to the same script skip
compilation. By default, the cache is stored under ``.cache/`` in the root of
the current Git repository. If Tilus is used outside a Git repository, the
default falls back to ``~/.cache/tilus``.

You can override the cache directory with :func:`tilus.option.cache_dir`:

.. code-block:: python

    import tilus
    tilus.option.cache_dir('./my-cache')

The cache directory can be safely deleted at any time --- Tilus will simply
recompile on the next run.


Viewing the Generated CUDA Kernel
----------------------------------

To inspect the generated CUDA source for a kernel, set the cache directory and
run your program:

.. code-block:: python

    import tilus
    tilus.option.cache_dir('./cache')

    # ... your program ...

After execution, the generated CUDA source can be found at::

    cache/programs/<program-hash>/module/source.cu


Dumping Intermediate Representations
------------------------------------

To inspect the intermediate representations (IRs) produced during compilation,
enable :func:`tilus.option.debug.dump_ir`:

.. code-block:: python

    import tilus
    tilus.option.cache_dir('./cache')
    tilus.option.debug.dump_ir()

    # ... your program ...

This writes the IR after each compiler pass into the cache directory:

- ``cache/programs/<program-hash>/ir/`` --- Tilus IR after each pass
- ``cache/programs/<program-hash>/module/ir/`` --- Hidet IR after each pass


Cache Structure
---------------

::

    cache/
    ├── scripts/                              # one entry per tilus script
    │   └── <script-name>/                    # class name, in snake_case
    │       └── <script-hash>/
    │           ├── programs/
    │           │   ├── 0 -> ../../programs/…  # symlink to the 1st schedule
    │           │   ├── 1 -> ../../programs/…  # symlink to the 2nd schedule
    │           │   └── ...
    │           └── dispatch_table.txt         # maps dynamic input sizes to programs
    │
    └── programs/                             # one entry per compiled program
        └── <program-hash>/                   # hex256 hash of program.txt + options.txt
            ├── program.txt                   # human-readable Tilus program
            ├── options.txt                   # compilation options
            ├── ir/                           # Tilus IR dumps (when dump_ir is enabled)
            └── module/
                ├── ir/                       # Hidet IR dumps (when dump_ir is enabled)
                ├── source.cu                 # generated CUDA source
                ├── compile.sh                # nvcc compilation command
                └── lib.so                    # compiled shared library

A **script** is a kernel template with a tuning space. Each **program** is a
concrete instantiation of a script with a specific schedule (tile sizes,
pipeline depths, etc.). The ``dispatch_table.txt`` records which program to
use for each combination of dynamic input sizes.
