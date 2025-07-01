Welcome to tilus's documentation!
=================================

**Tilus** is a domain-specific language (DSL) for GPU programming, designed with:

* Thread-block-level granularity and tensors as the core data type
* Explicit control over shared memory and tensor layouts (unlike Triton)
* Support for low-precision types with arbitrary bit-widths

Additional features include automatic tuning, caching, and a Pythonic interface for ease of use.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting-started/install
   getting-started/tutorials/__init__

.. toctree::
   :maxdepth: 1
   :caption: Programming Guides

   programming-guides/overview
   programming-guides/tilus-script
   programming-guides/type-system/__init__
   programming-guides/instructions
   programming-guides/control-flow
   programming-guides/cache
   programming-guides/autotuning
   programming-guides/low-precision-support

.. toctree::
   :maxdepth: 1
   :caption: Python API

   python-api/tilus
   python-api/tilus-script
   python-api/tilus-lang-attributes
   python-api/tilus-ir/__index__
