Getting started with Syncopy
============================

Installing Syncopy
------------------

Until the end of our public beta, Syncopy is only hosted on
`TestPyPI <https://test.pypi.org/project/syncopy/>`_. It can be installed using
`Pip <https://pypi.org/project/pip/>`_:

.. code-block:: bash

    pip install -i https://test.pypi.org/simple/ esi-syncopy

Eventually, Syncopy will be hosted on `PyPI <https://pypi.org/>`_ and
`conda-forge <https://conda-forge.org/>`_. If you're working on the ESI cluster
installing Syncopy is only necessary if you create your own Conda environment.

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
this folder. On the `ESI JupyterHub <https://jupyterhub.esi.local>`_, the
variable is set to use the high performance storage:

.. code-block:: bash

    SPYTMPDIR=/mnt/hpx/home/$USER/.spy


Importing Syncopy
-----------------

To start using Syncopy you have to import it in your Python code:

.. code-block:: python

    import syncopy as spy

All :doc:`user-facing functions and classes <user/user_api>` can then be
accessed with the ``spy.`` prefix, e.g.

.. code-block:: python

    spy.load("~/testdata.spy")


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
