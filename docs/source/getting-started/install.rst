Installation
============

To install ``tilus``, you can use pip:

.. code-block:: bash

    pip install tilus


.. note::

    Tilus depends on `cuda-python`. If your GPU driver is older than **580.65.06**, you will need to install an older version of cuda-python to ensure compatibility.

    .. code-block:: bash

        pip install tilus "cuda-python<13"

If you want to install the latest development version, you can clone the repository and install it from source:

.. code-block:: bash

    git clone git@github.com:NVIDIA/tilus.git
    cd tilus
    pip install -e .

The ``-e`` option means "editable", so you can modify the source code and see the changes immediately without reinstalling.
