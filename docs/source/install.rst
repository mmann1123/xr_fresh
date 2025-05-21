Installation
===================

Installation
------------

To install ``xr_fresh``, you can use pip. However, since ``xr_fresh`` includes a C++ extension module, it requires compilation during the installation process. Here are the steps to install ``xr_fresh``:

Prerequisites
~~~~~~~~~~~~~

- Conda or mamba (`Instructions here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_)
- C++ compiler (e.g., g++ on Linux, clang on macOS, or MSVC on Windows)

Linux, OSx & Windows Install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # add dependency
   conda create -n xr_fresh geowombat -c conda-forge
   conda activate xr_fresh
   # clone repository
   cd # to desired location
   git clone https://github.com/mmann1123/xr_fresh
   cd xr_fresh
   pip install -U pip setuptools wheel
   pip install .

.. note::

   If you run into problems related to ``rle`` try running ``python setup.py build_ext --inplace`` from the ``xr_fresh`` directory.

To run PCA you must also install ``ray``:

.. code-block:: bash

   conda install -c conda-forge "ray-default"

.. note::

   ``ray`` is only in beta for Windows and will not be installed by default. Please read more about the installation `here <https://docs.ray.io/en/latest/ray-overview/installation.html>`_.