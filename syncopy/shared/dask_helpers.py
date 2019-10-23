# -*- coding: utf-8 -*-
# 
# Helper routines for working w/dask 
# 
# Created: 2019-05-22 12:38:16
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-10-23 15:36:21>

# Builtin/3rd party package imports
import os
import sys
import socket
import subprocess
import getpass
import time
import inspect
import numpy as np
from tqdm import tqdm
if sys.platform == "win32":
    # tqdm breaks term colors on Windows - fix that (tqdm issue #446)
    import colorama
    colorama.deinit()
    colorama.init(strip=False)

# Local imports
from syncopy import __dask__
from syncopy.shared.parsers import scalar_parser, io_parser
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYIOError
from syncopy.shared.queries import user_input, user_yesno
if __dask__:
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, get_client
    from datetime import datetime, timedelta

__all__ = ["esi_cluster_setup", "cluster_cleanup"]


# Setup SLURM cluster
def esi_cluster_setup(partition="8GBS", n_jobs=2, mem_per_job=None,
                      timeout=180, interactive=True, start_client=True,
                      **kwargs):
    """
    Start a distributed Dask cluster of parallel processing workers using SLURM 
    (or local multi-processing)
    
    Parameters
    ----------
    partition : str
        Name of SLURM partition/queue to use
    n_jobs : int
        Number of jobs to spawn
    mem_per_job : None or str
        Memory booking for each job. Can be specified either in megabytes 
        (e.g., ``mem_per_job = 1500MB``) or gigabytes (e.g., ``mem_per_job = "2GB"``). 
        If `mem_per_job` is `None`, it is attempted to infer a sane default value
        from the chosen queue, e.g., for ``partition = "8GBS"`` `mem_per_job` is 
        automatically set to the allowed maximum of `'8GB'`. However, even in
        queues with guaranted memory bookings, it is possible to allocate less
        memory than the allowed maximum per job to spawn numerous low-memory 
        jobs. See Examples for details. 
    timeout : int
        Number of seconds to wait for requested jobs to start up. 
    interactive : bool
        If `True`, user input is required in case not all jobs could 
        be started in the provided waiting period (determined by `timeout`). 
        If `interactive` is `False` and the jobs could not be started
        within `timeout` seconds, a `TimeoutError` is raised. 
    start_client : bool
        If `True`, a distributed computing client is launched and attached to
        the workers. If `start_client` is `False`, only a distributed 
        computing cluster is started to which compute-clients can connect. 
    **kwargs : dict
        Additional keyword arguments can be used to control job-submission details. 
        
    Returns
    -------
    proc : object
        A distributed computing client (if ``start_client = True``) or 
        a distributed computing cluster (otherwise). 

    Examples
    --------
    The following command launches 10 SLURM jobs with 2 gigabytes memory each 
    in the `8GBS` partition
    
    >>> spy.esi_cluster_setup(n_jobs=10, partition="8GBS", mem_per_job="2GB") 
    
    If you want to access properties of the created distributed computing client, 
    assign an explicit return quantity, i.e., 
    
    >>> client = spy.esi_cluster_setup(n_jobs=10, partition="8GBS", mem_per_job="2GB") 
    
    The underlying distributed computing cluster can be accessed using
    
    >>> client.cluster
    
    Notes
    -----
    Syncopy's parallel computing engine relies on the concurrent processing library
    `Dask <https://docs.dask.org/en/latest/>`_. Thus, the distributed computing
    clients used by Syncopy are in fact instances of :class:`dask.distributed.Client`. 
    This function specifically acts  as a wrapper for :class:`dask_jobqueue.SLURMCluster`. 
    Users familiar with Dask in general and its distributed scheduler and cluster 
    objects in particular, may leverage Dask's entire API to fine-tune parallel 
    processing jobs to their liking (if wanted). 
    
    See also
    --------
    cluster_cleanup : remove dangling parallel processing job-clusters
    """
    
    # For later reference: dynamically fetch name of current function
    funcName = "Syncopy <{}>".format(inspect.currentframe().f_code.co_name)

    # Retrieve all partitions currently available in SLURM
    out, err = subprocess.Popen("sinfo -h -o %P",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, shell=True).communicate()
    if len(err) > 0:
        
        # SLURM is not installed, either allocate `LocalCluster` or just leave
        if "sinfo: not found" in err:
            msg = "{name:s} SLURM does not seem to be installed on this machine " +\
                  "({host:s}). Do you want to start a local multi-processing " +\
                  "computing client instead? "
            if user_yesno(msg.format(name=funcName, host=socket.gethostname()), default="no"):
                client = Client()
                if start_client:
                    return client
                return client.cluster
            return 

        # SLURM is installed, but something's wrong        
        msg = "SLURM queuing system from node {node:s}. " +\
              "Original error message below:\n{error:s}"
        raise SPYIOError(msg.format(node=socket.gethostname(), error=err))
    options = out.split()

    # Make sure we're in a valid partition (exclude IT partitions from output message)
    if partition not in options:
        valid = list(set(options).difference(["DEV", "PPC"]))
        raise SPYValueError(legal="'" + "or '".join(opt + "' " for opt in valid),
                            varname="partition", actual=partition)

    # Parse job count
    try:
        scalar_parser(n_jobs, varname="n_jobs", ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        raise exc

    # Get requested memory per job
    if mem_per_job is not None:
        if not isinstance(mem_per_job, str):
            raise SPYTypeError(mem_per_job, varname="mem_per_job", expected="string")
        if not any(szstr in mem_per_job for szstr in ["MB", "GB"]):
            lgl = "string representation of requested memory (e.g., '8GB', '12000MB')"
            raise SPYValueError(legal=lgl, varname="mem_per_job", actual=mem_per_job)

    # Query memory limit of chosen partition and ensure that `mem_per_job` is
    # set for partitions w/o limit
    idx = partition.find("GB")
    if idx > 0:
        mem_lim = int(partition[:idx])
    else:
        if partition == "PREPO":
            mem_lim = 16
        else:
            if mem_per_job is None:
                lgl = "explicit memory amount as required by partition '{}'"
                raise SPYValueError(legal=lgl.format(partition),
                                    varname="mem_per_job", actual=mem_per_job)
        mem_lim = np.inf

    # Consolidate requested memory with chosen partition (or assign default memory)
    if mem_per_job is None:
        mem_per_job = str(mem_lim) + "GB"
    else:
        if "MB" in mem_per_job:
            mem_req = round(int(mem_per_job[:mem_per_job.find("MB")]) / 1000, 1)
            if int(mem_req) == mem_req:
                mem_req = int(mem_req)
        else:
            mem_req = int(mem_per_job[:mem_per_job.find("GB")])
        if mem_req > mem_lim:
            msg = "{name:s} WARNING: `mem_per_job` exceeds limit of " +\
                  "{lim:d}GB for partition {par:s}. Capping memory at partition limit. "
            print(msg.format(name=funcName, lim=mem_lim, par=partition))
            mem_per_job = str(int(mem_lim)) + "GB"

    # Parse requested timeout period
    try:
        scalar_parser(timeout, varname="timeout", ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        raise exc

    # Determine if cluster allocation is happening interactively
    if not isinstance(interactive, bool):
        raise SPYTypeError(interactive, varname="interactive", expected="bool")

    # Determine if a dask client was requested
    if not isinstance(start_client, bool):
        raise SPYTypeError(start_client, varname="start_client", expected="bool")

    # Set/get "hidden" kwargs
    workers_per_job = kwargs.get("workers_per_job", 1)
    try:
        scalar_parser(workers_per_job, varname="workers_per_job",
                      ntype="int_like", lims=[1, 8])
    except Exception as exc:
        raise exc

    n_cores = kwargs.get("n_cores", 1)
    try:
        scalar_parser(n_cores, varname="n_cores",
                      ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        raise exc

    slurm_wdir = kwargs.get("slurmWorkingDirectory", None)
    if slurm_wdir is None:
        usr = getpass.getuser()
        slurm_wdir = "/mnt/hpx/slurm/{usr:s}/{usr:s}_{date:s}"
        slurm_wdir = slurm_wdir.format(usr=usr,
                                       date=datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.makedirs(slurm_wdir, exist_ok=True)
    else:
        try:
            io_parser(slurm_wdir, varname="slurmWorkingDirectory", isfile=False)
        except Exception as exc:
            raise exc

    # Create `SLURMCluster` object using provided parameters
    out_files = os.path.join(slurm_wdir, "slurm-%j.out")
    cluster = SLURMCluster(cores=n_cores,
                           memory=mem_per_job,
                           processes=workers_per_job,
                           local_directory=slurm_wdir,
                           queue=partition,
                           name="spycluster",
                           job_extra=["--output={}".format(out_files)])
                           # interface="asdf", # interface is set via `psutil.net_if_addrs()`
                           # job_extra=["--hint=nomultithread",
                           #            "--threads-per-core=1"]

    # Compute total no. of workers and up-scale cluster accordingly
    total_workers = n_jobs * workers_per_job
    cluster.scale(total_workers)

    # Fire up waiting routine to avoid premature cluster setups
    if _cluster_waiter(cluster, funcName, total_workers, timeout, interactive):
        return
    
    # Kill a zombie cluster in non-interactive mode
    if not interactive and cluster._count_active_workers() == 0:
        cluster.close()
        err = "SLURM jobs could not be started within given time-out " +\
              "interval of {0:d} seconds"
        raise TimeoutError(err.format(timeout))
    
    # Highlight how to connect to dask performance monitor
    print("{name:s} Cluster dashboard accessible at {dash:s}".format(name=funcName,
                                                                     dash=cluster.dashboard_link))

    # If client was requested, return that instead of the created cluster
    if start_client:
        return Client(cluster)
    return cluster


def _cluster_waiter(cluster, funcName, total_workers, timeout, interactive):
    """
    Local helper that can be called recursively
    """

    # Wait until all workers have been started successfully or we run out of time
    wrkrs = cluster._count_active_workers()
    to = str(timedelta(seconds=timeout))[2:]
    fmt = "{desc}: {n}/{total} \t[elapsed time {elapsed} | timeout at " + to + "]"
    ani = tqdm(desc="{} SLURM workers ready".format(funcName), total=total_workers,
               leave=True, bar_format=fmt, initial=wrkrs)
    counter = 0
    while cluster._count_active_workers() < total_workers and counter < timeout:
        time.sleep(1)
        counter += 1
        ani.update(max(0, cluster._count_active_workers() - wrkrs))
        wrkrs = cluster._count_active_workers()
        ani.refresh()   # force refresh to display elapsed time every second
    ani.close()

    # If we ran out of time before all workers could be started, ask what to do
    if counter == timeout and interactive:
        msg = "{name:s} SLURM jobs could not be started within given time-out " +\
              "interval of {time:d} seconds"
        print(msg.format(name=funcName, time=timeout))
        query = "{name:s} Do you want to [k]eep waiting for 60s, [a]bort or " +\
                "[c]ontinue with {wrk:d} workers?"
        choice = user_input(query.format(name=funcName, wrk=wrkrs), valid=["k", "a", "c"])

        if choice == "k":
            return _cluster_waiter(cluster, funcName, total_workers, 60, True)
        elif choice == "a":
            print("{} Closing cluster...".format(funcName))
            cluster.close()
            return True
        else:
            if wrkrs == 0:
                query = "{} Cannot continue with 0 workers. Do you want to " +\
                        "[k]eep waiting for 60s or [a]bort?"
                choice = user_input(query.format(funcName), valid=["k", "a"])
                if choice == "k":
                    _cluster_waiter(cluster, funcName, total_workers, 60, True)
                else:
                    print("{} Closing cluster...".format(funcName))
                    cluster.close()
                    return True

    return False

def cluster_cleanup():
    """
    Stop and close dangling parallel processing jobs
    
    Parameters
    ----------
    Nothing : None
    
    Returns
    -------
    Nothing : None
    
    See also
    --------
    esi_cluster_setup : Launch a SLURM jobs on the ESI compute cluster
    """
    
    # For later reference: dynamically fetch name of current function
    funcName = "Syncopy <{}>".format(inspect.currentframe().f_code.co_name)
    
    # Attempt to establish connection to dask client
    try:
        client = get_client()
    except ValueError:
        print("{} WARNING: No dangling clients or clusters found.".format(funcName))
        return
    except Exception as exc:
        raise exc
    
    # Prepare message for prompt
    userName = getpass.getuser()
    outDir = client.cluster.job_header.partition("--output=")[-1]
    jobID = outDir.partition("{}_".format(userName))[-1].split(os.sep)[0]
    userClust = "{0}_{1}".format(userName, jobID)
    nWorkers = client.cluster._count_active_and_pending_workers()
    
    # If connection was successful, first close the client, then the cluster
    client.close()
    client.cluster.close()
    
    # Communicate what just happened and get outta here
    msg = "{fname:s} Successfully shut down cluster {cname:s} " +\
          "containing {nj:d} workers"
    print(msg.format(fname=funcName,
                     nj=nWorkers,
                     cname=userClust))

    return
