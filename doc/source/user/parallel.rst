.. _parallel:

--------------------
Parallel Processing
--------------------

.. image:: /_static/dask_logo.png
   :height: 60px
   :target: https://dask.org

Syncopy employs `Dask <https://dask.org/>`_ as its parallel processing engine and it gets installed alongside Syncopy automatically. Dask is a powerful modern parallel computing framework, allowing concurrency on a variety of systems from a single laptop up to cloud infrastructure.

In general parallelization in Syncopy works by:

- setup a suitable Dask cluster
- initialize a Dask client connected to that cluster
- use Syncopy as usual

**Syncopy always checks if a Dask client is available prior to any computation** and automatically allocates compute tasks to the provided resources. In effect, on any Dask compatible compute system Syncopy can be run in parallel. The most common Dask compatible compute system at academic institutions is most likely a high performance computing (HPC) cluster with a job scheduler like Slurm, run by your institution's computing center, or a smaller system run by an individual lab that requires lots of computing power.

.. contents::
   :local:

Dask Setup and Integration
--------------------------

To quickly enable parallel processing on a local machine we can launch a  `LocalCluster <https://docs.dask.org/en/stable/deploying-python.html#localcluster>`_::

  import dask.distributed as dd

  # request 8 worker processes
  cluster = dd.LocalCluster(n_workers=8)

  # attach a client to the cluster
  client = dd.Client(cluster)

With the client in place, we can simply call a Syncopy function as usual::

  import syncopy as spy

  # create ~1GB of white noise
  data = spy.synthdata.white_noise(nTrials=5_000, nChannels=32)

  # compute the power spectra
  spec = spy.freqanalysis(data)

and the computations are automatically **parallelized over trials**. That means Syncopy code is the same for both parallel and sequential processing!

Syncopy notifies about parallel computations via logging:

.. code-block:: bash

     15:17:50 - IMPORTANT: 8/8 workers available, starting computation..
     15:17:51 - IMPORTANT: ..attaching to running Dask client:
	<Client: 'tcp://127.0.0.1:58228' processes=8 threads=8, memory=16.00 GiB>

Dask itself offers a `dashboard <https://docs.dask.org/en/stable/dashboard.html>`_ which runs in the browser, checkout http://localhost:8787/status to see the status of the ``LocalCluster`` we created.

.. note::
   Once a Dask client/cluster is alive, it can be re-used for any number of computations with Syncopy. Don't start a second cluster for a new computation!

.. note::
   The Dask setup is independent of any Syncopy code. It might be a good idea to create a standalone ``setup_my_dask.py`` script which can be executed once before running any number of analyses with Syncopy.

.. warning::
   Without a **prior initialization** of a Dask client all computations in Syncopy are only executed sequentially, relying solely on low-level built-in parallelization offered by external libraries like `NumPy <https://numpy.org/>`_.


Example Dask SLURM cluster Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are working on a HPC system, remember you need to configure your own Dask cluster **before** running any Syncopy
computations, and Syncopy will happily use the provided ressoures. See the `Dask tutorial <https://tutorial.dask.org/>`_
for a general introduction about allocating distributed computing ressources. On a SLURM cluster, a basic setup
could look like this::

  import dask.distributed as dd
  import dask_jobqueue as dj

  slurm_wdir = "/path/to/workdir/"
  n_jobs = 8
  reqMem = 32
  queue = 'slurm_queue'

  cluster = dj.SLURMCluster(cores=1, memory=f'{reqMem} GB', processes=1,
                            local_directory=slurm_wdir,
                            queue=queue)

  cluster.scale(n_jobs)
  client = dd.Client(cluster)

  # now start syncopy computations
  # global dask `client` gets automatically recognized

With a client connected to a Dask cluster in place, we can run computations with Syncopy as usual::

  import syncopy as spy

  data = spy.synthdata.white_noise(nTrials=500, nSamples=10_000, nChannels=10)

  # band pass filtering between 20Hz and 40Hz
  spec = spy.preprocessing(data, freq=[20, 40], filter_type='bp')

If the Dask clurm cluster was freshly requested, we first have to wait until at least 1 worker is ready:

.. code-block:: bash

   Syncopy <check_workers_available> INFO: 0/8 workers available, waiting.. 0s
   Syncopy <check_workers_available> INFO: 0/8 workers available, waiting.. 2s
   Syncopy <check_workers_available> INFO: 0/8 workers available, waiting.. 4s
   Syncopy <check_workers_available> INFO: 3/8 workers available, waiting.. 6s
   Syncopy <parallel_client_detector> INFO: ..attaching to running Dask client:
   <Client: 'tcp://10.100.32.3:42673' processes=3 threads=3, memory=92.40 GiB>
   [###################################     ] | 88% Completed | 52.3

Syncopy employs a timeout of 360s (6 minutes), if after that time not a single worker is available the computations get aborted.

To check the status of the Dask cluster manualy you can do::

  dd.get_client()

This will output the current state of the client/cluster:

.. code-block:: bash

  >>> <Client: 'tcp://10.100.32.3:42673' processes=3 threads=3, memory=92.40 GiB>

indicating here that 3 workers are available at this very moment.

.. hint::
   For a basic introduction to HPC computing see this `wiki <https://hpc-wiki.info>`_
   and/or the Slurm `documentation <https://slurm.schedmd.com/>`_.


Channel Parallelisation
------------------------

Standard parallelization is over trials, additional parallelization over channels can be achieved by using the `chan_per_worker` keyword:

.. code-block:: python

    spec = spy.freqanalysis(data,
		            method="mtmfft",
			    foilim=[1, 150],
			    tapsmofrq=10,
			    parallel=True,
			    chan_per_worker=40)

This would allocate the computation for each trial and 40 channel chunk to an independent computing process. Note that the number of parallel processes is generally limited, depending on the computing resources available. Hence setting ``chan_per_worker=1`` can be actually quite inefficient when the data has say 200 channels but only 4 parallel processes are available at any given time. In general, if there are only few trials, it is safe and even recommended to set `chan_per_worker` to a fairly low number. On the other hand, depending on the HPC setup at hand, being to greedy here might also spawn a lot of jobs and hence might induce long waiting times.


ESI: ACME - Cluster Setup
--------------------------

If you are on the `ESI <https://www.esi-frankfurt.de/>`_ HPC, the Dask cluster setup can be
handled by :func:`~acme.esi_cluster_setup` which is available when `ACME <https://github.com/esi-neuroscience/acme>`_ is installed.
It provides a convenient way to launch a custom-tailored cluster of parallel SLURM workers on the ESI HPC.

As with any other Dask cluster
:func:`~acme.esi_cluster_setup` has to be called **before** launching the actual calculation.
For example::

    spyClient = spy.esi_cluster_setup(partition="16GBXL", n_jobs=10)

starts 10 concurrent SLURM workers in the `16GBXL` queue if run on the ESI HPC
cluster. As usual all subsequent invocations of Syncopy analysis routines will automatically
pick up the ``spyClient`` and distribute any occurring computational payload across
the workers collected in ``spyClient``.
