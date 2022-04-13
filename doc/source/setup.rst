Install Syncopy
===============

Syncopy can be installed using `conda <https://anaconda.org>`_:

.. code-block:: bash

    conda install -c conda-forge esi-syncopy

Alternatively it is also available on `Pip <https://pypi.org/project/pip/>`_:

.. code-block:: bash

    pip install esi-syncopy

If you're working on the ESI cluster installing Syncopy is only necessary if
you create your own Conda environment.

.. _install_acme:

Installing parallel processing engine ACME
--------------------------------------------

To harness the parallel processing capabilities of Syncopy
it is necessary to install `ACME <https://github.com/esi-neuroscience/acme>`_.

Again either via conda

.. code-block:: bash

    conda install -c conda-forge esi-acme

or pip

.. code-block:: bash

    pip install esi-acme


Importing Syncopy
-----------------

To start using Syncopy you have to import it in your Python code:

.. code-block:: python

    import syncopy as spy

All :doc:`user-facing functions and classes <user/user_api>` can then be
accessed with the ``spy.`` prefix, e.g.

.. code-block:: python

    spy.load("~/testdata.spy")

.. _start_parallel:

Starting Up Parallel Workers
----------------------------

In Syncopy all computations are designed to run in parallel taking advantage of
modern multi-core system architectures. The simplest way to leverage any available
concurrent processing hardware is to use the `parallel` keyword, e.g.,

.. code-block:: python

    spy.freqanalysis(data, method="mtmfft", parallel=True)

This will allocate a parallel worker for each trial defined in `data`. If your code
is running on the ESI cluster, Syncopy will automatically use the existing SLURM
scheduler, in a single-machine setup, any available local multi-processing resources
will be utilized. More details can be found in the :doc:`Data Analysis Guide <user/processing>`

.. _setup_env:

Setting Up Your Python Environment
----------------------------------

On the ESI cluster, ``/opt/conda/envs/syncopy`` provides a
pre-configured and tested Conda environment with the most recent Syncopy
version. This environment can be easily started using the `ESI JupyterHub
<https://jupyterhub.esi.local>`_

Syncopy makes heavy use of temporary files, which may become large (> 100 GB).
The storage location can be set using the `environmental variable
<https://linuxhint.com/bash-environment-variables/>`_ :envvar:`SPYTMPDIR`, which
by default points to your home directory:

.. code-block:: bash

    SPYTMPDIR=~/.spy

The performance of Syncopy strongly depends on the read and write speed in
this folder. On the ESI cluster, the variable is set to use the high performance
storage:

.. code-block:: bash

    SPYTMPDIR=/cs/home/$USER/.spy
