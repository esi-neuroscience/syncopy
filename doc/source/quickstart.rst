Getting started with Syncopy
============================

Installing Syncopy
------------------

Until the end of the early alpha development phase, Syncopy is only hosted on
`TestPyPI <https://test.pypi.org/project/syncopy/>`_. It can be installed using
`Pip <https://pypi.org/project/pip/>`_:

.. code-block:: bash

    pip install -i https://test.pypi.org/simple/ syncopy

Eventually, Syncopy will be hosted on `PyPI <https://pypi.org/>`_ and
`conda-forge <https://conda-forge.org/>`_. If you're working on the ESI cluster
installing Syncopy is only necessary if you create your own environment.

Setting up your Python environment
----------------------------------

On the ESI cluster, ``/opt/ESIsoftware/python/envs/syncopy`` provides a
pre-configured and tested Conda environment with the most recent Syncopy
version. This environment can be easily started using the `ESI JupyterHub
<https://jupyterhub.esi.local>`_

Syncopy makes heavy use of temporary files, which may become large (> 100 GB).
The storage location can be set using the `environmental variable
<https://linuxhint.com/bash-environment-variables/>`_ :envvar:`SPYTMPDIR`, which
by default points to your home directory:

.. code-block:: bash

    SPYTMPDIR=~/.spy

The performance of Syncopy is strongly depends on the read and write speed into
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


Starting up parallel workers
----------------------------

In Syncopy all computations are designed to run in parallel taking advantage of
modern multi-core system architectures. However, before any computation you
have to initialize your "cluster" of parallel workers, which are the CPU cores
if run on a single machine (laptop or workstation) or compute jobs if run on a
computing cluster (e.g. SLURM).

To initialize the cluster, use the :func:`~syncopy.esi_cluster_setup` function.
For example,

.. code-block:: python

    spy.esi_cluster_setup(n_jobs=10)

will start 10 parallel workers on the SLURM cluster of the ESI. If the
:func:`~syncopy.esi_cluster_setup` function is not run before any computation,
all following computations will be computed sequentially.
