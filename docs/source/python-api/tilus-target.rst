tilus.target
============

Tilus automatically detects the GPU at runtime and selects the appropriate compilation target.
You can query or override the target using the functions below.

.. autosummary::
    :toctree: generated/

    tilus.target.get_current_target
    tilus.target.set_current_target

Predefined Targets
------------------

Each target represents a GPU architecture with its compute capability and feature suffix.
Tilus uses these to determine which instructions and optimizations are available.

.. list-table::
   :header-rows: 1
   :widths: 25 15 30

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
   * - ``nvgpu_sm90`` / ``nvgpu_sm90a``
     - 9.0
     - H100, H200
   * - ``nvgpu_sm100`` / ``nvgpu_sm100a``
     - 10.0
     - B200, GB200

**Feature suffixes:**

- No suffix: base architecture (instructions available on all chips with that SM version)
- ``a``: architecture-specific (e.g., ``sm_90a`` for Hopper-only features like WGMMA)
- ``f``: family variant

Usage
-----

.. code-block:: python

    import tilus
    from tilus.target import set_current_target, nvgpu_sm90a

    # Query the auto-detected target
    target = tilus.get_current_target()
    print(target)  # e.g., nvgpu/sm90a

    # Override the target (e.g., for cross-compilation)
    set_current_target(nvgpu_sm90a)
