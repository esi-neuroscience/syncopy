.. _parallel:

------------------------------
Serial and Parallel Processing
------------------------------

.. contents:: Topics covered
   :local:

By default, all computations in Syncopy are executed sequentially relying solely 
on low-level built-in parallelization offered by external libraries like `NumPy <https://numpy.org/>`_. 
The simplest way to enable full concurrency for a given Syncopy calculation 
is by using the `parallel` keyword supported by all Syncopy meta-functions, i.e., 

.. code-block:: python
      
    spec = spy.freqanalysis(data, method="mtmfft", foilim=[1, 150], tapsmofrq=10, parallel=True)

or 

.. code-block:: python
      
    cfg = spy.get_defaults(spy.freqanalysis)
    cfg.method = 'mtmfft'
    cfg.foilim = [1, 150]
    cfg.tapsmofrq = 10
    cfg.parallel = True
    spec = spy.freqanalysis(cfg, data)

Default parallelization is over trials, additional parallelization over channels can be achieved by using the `chan_per_worker` keyword:

.. code-block:: python

    spec = spy.freqanalysis(data,
		            method="mtmfft",
			    foilim=[1, 150],
			    tapsmofrq=10,
			    parallel=True,
			    chan_per_worker=40)

This would allocate the computation for each trial and 40 channel chunk to an independent computing process. Note that the number of parallel processes is generally limited, depending on the computing resources available. Hence setting ``chan_per_worker=1`` can be actually quite inefficient when the data has say 200 channels but only 4 parallel processes are available at any given time. In general, if there are only few trials, it is safe and even recommended to set `chan_per_worker` to a fairly low number. On the other hand, depending on the compute cluster setup, being to greedy here might also spawn a lot of jobs and hence might induce long waiting times. 

    
ACME - Cluster Setup
~~~~~~~~~~~~~~~~~~~~
    
More fine-grained control over allocated resources and load-balancer options is available 
via the routine :func:`~syncopy.esi_cluster_setup` which is available when
`ACME <https://github.com/esi-neuroscience/acme>`_ is installed.
It provides a convenient way to launch a custom-tailored 
"cluster" of parallel workers (compute jobs if run on a cluster computing manager such as SLURM)
on the ESI HPC. 
Thus, instead of simply "turning on" parallel computing via a keyword and letting 
Syncopy choose an optimal setup for the computation at hand, more fine-grained 
control over resource allocation and management can be achieved via running 
:func:`~syncopy.esi_cluster_setup` **before** launching the actual calculation. 
For example::

    spyClient = spy.esi_cluster_setup(partition="16GBXL", n_jobs=10)

starts 10 concurrent SLURM workers in the `16GBXL` queue if run on the ESI HPC 
cluster. All subsequent invocations of Syncopy analysis routines will automatically 
pick up ``spyClient`` and distribute any occurring computational payload across 
the workers collected in ``spyClient``. 

.. hint::

   If `esi_cluster_setup` is unavailable, have a look at :ref:`install_acme` For general deployment on other HPC systems please contact the ACME
   team directly.

Manual Dask cluster Setup
~~~~~~~~~~~~~~~~~~~~~~~~~

With ``parallel=True`` Syncopy looks for a running `Dask <https://dask.org/>`_ client to attach to,
and if none is found a Dask ``LocalCluster`` is started as a fallback to allow basic parallel execution on single machines.

If you are working on a HPC system, you can configure your own Dask cluster **before** running any Syncopy
computations, and Syncopy will happily use the provided ressoures. See the `Dask tutorial <https://tutorial.dask.org/>`_
for a general introduction about allocating distributed computing ressources. On a SLURM cluster, a basic setup
could look like this::

  import dask.distributed as dd
  import dask_jobqueue as dj
  import syncopy as spy
  
  slurm_wdir = "/path/to/workdir/"
  n_jobs = 8
  reqMem = 32  
  queue = 'slurm_queue'

  cl = dj.SLURMCluster(cores=1, memory=f'{reqMem} GB', processes=1,
                       local_directory=slurm_wdir,
                       queue=queue)

  cl.scale(n_jobs)
  client = dd.Client(cl)

  # now start syncopy computations,
  # global dask `client` gets automatically recognized
  # no need for `parallel=True`
  spy.freqanalysis(...)

If the Dask clurm cluster was freshly requested, we first have to wait until all workers are ready:

.. code-block:: bash

   Syncopy <check_workers_available> INFO: 0/8 workers available, waiting.. 0s
   Syncopy <check_workers_available> INFO: 3/8 workers available, waiting.. 2s
   Syncopy <check_workers_available> INFO: 7/8 workers available, waiting.. 4s
   Syncopy <parallel_client_detector> INFO: ..attaching to running Dask client:
   <Client: 'tcp://10.100.32.3:42673' processes=8 threads=8, memory=238.40 GiB>
   [###################################     ] | 88% Completed | 52.3

.. hint::
   For a basic introduction to HPC computing see this `wiki <https://hpc-wiki.info>`_
   and/or the Slurm `documentation <https://slurm.schedmd.com/>`_.
