tilus.target
============

Tilus automatically detects the GPU at runtime and selects the most capable
compilation target for the installed hardware. You can query or override the
target using the functions below.

.. autosummary::
    :toctree: generated/

    tilus.target.get_current_target
    tilus.target.set_current_target

Predefined Targets
------------------

Each target represents a GPU architecture with its compute capability and
feature suffix. Tilus uses these to determine which instructions and
optimizations are available.

.. list-table::
   :header-rows: 1
   :widths: 30 10 30

   * - Target
     - SM
     - GPU Examples
   * - ``nvgpu_sm70``
     - 7.0
     - V100
   * - ``nvgpu_sm75``
     - 7.5
     - T4, RTX 2080
   * - ``nvgpu_sm80``
     - 8.0
     - A100
   * - ``nvgpu_sm86``
     - 8.6
     - RTX 3090, A40
   * - ``nvgpu_sm89``
     - 8.9
     - L4, L40, RTX 4090
   * - ``nvgpu_sm90`` / ``sm90a``
     - 9.0
     - H100, H200
   * - ``nvgpu_sm100`` / ``sm100f`` / ``sm100a``
     - 10.0
     - B200, GB200
   * - ``nvgpu_sm103`` / ``sm103f`` / ``sm103a``
     - 10.3
     - GB300
   * - ``nvgpu_sm120`` / ``sm120f`` / ``sm120a``
     - 12.0
     - RTX 5080, RTX 5090

Feature Suffixes
----------------

Starting with SM 9.0, NVIDIA targets can carry a **feature suffix** that
controls which architecture-conditional features are enabled. Understanding
the suffixes is important when writing kernels that use advanced hardware
features (e.g., Tensor Memory, tcgen05 MMA, TMA).

**No suffix:**
The base architecture (e.g., ``sm_100``). Only instructions guaranteed on
*every* chip with that compute capability are available. Use this when you
only need baseline capabilities and want maximum hardware compatibility.

**`a` (architecture-specific):**
The full-featured variant for that exact architecture (e.g., ``sm_100a``).
Enables all architecture-conditional features, such as Tensor Memory
allocation modes, tcgen05 Tensor Core operations, and special TMA behaviors.
Use this when you want every available hardware feature and are targeting a
specific GPU.

**`f` (family-portable):**
The family-profile variant (e.g., ``sm_100f``). Enables features that are
portable across all implementations within the same SM family. A feature
supported on ``sm_100f`` is guaranteed on any chip in the ``sm_100`` family
that advertises ``f``-level support. Use this when you want advanced features
with portability across family variants.

The compatibility relationship is:

- ``sm_100a`` supports everything in ``sm_100f`` and ``sm_100``.
- ``sm_100f`` supports everything in ``sm_100``, but not ``sm_100a``-only features.
- ``sm_100`` supports only baseline features.

.. note::

   When Tilus auto-detects the target, it picks the most capable variant
   available: ``a`` first, then ``f``, then the base. For example, on a B200
   GPU (SM 10.0), Tilus selects ``nvgpu_sm100a``.

Usage
-----

.. code-block:: python

    import tilus
    from tilus.target import set_current_target, nvgpu_sm100a

    # Query the auto-detected target
    target = tilus.get_current_target()
    print(target)  # e.g., nvgpu/sm100a

    # Override the target (e.g., for cross-compilation)
    set_current_target(nvgpu_sm100a)
