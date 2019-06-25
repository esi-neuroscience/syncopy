# -*- coding: utf-8 -*-
#
# Helper routines for working w/dask 
# 
# Created: 2019-05-22 12:38:16
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-06-25 17:04:31>

# Builtin/3rd party package imports
import os
import sys
import socket
import subprocess
import getpass
import time
import numpy as np
from dask_jobqueue import SLURMCluster
from datetime import datetime
from tqdm import tqdm
if sys.platform == "win32":
    # tqdm breaks term colors on Windows - fix that (tqdm issue #446)
    import colorama
    colorama.deinit()
    colorama.init(strip=False)

# Local imports
from syncopy.shared import scalar_parser, io_parser
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYIOError
from syncopy.shared.queries import user_input

__all__ = ["esi_cluster_setup"]


# Setup SLURM cluster
def esi_cluster_setup(partition="8GBS", n_jobs=2, mem_per_job=None,
                      timeout=180, **kwargs):
    """
    Coming soon(ish)
    """

    # Retrieve all partitions currently available in SLURM
    out, err = subprocess.Popen("sinfo -h -o %P",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, shell=True).communicate()
    if len(err) > 0:
        msg = "SLURM queuing system from node {}".format(socket.gethostname())
        raise SPYIOError(msg)
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
            mem_lim = "16GB"
        else:
            if mem_per_job is None:
                lgl = "explicit memory amount as required by partition '{}'"
                raise SPYValueError(legal=lgl.format(partition),
                                    varname="mem_per_job", actual=mem_per_job)

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
            msg = "<esi_cluster_setup> WARNING: `mem_per_job` exceeds limit of " +\
                  "{lim:d}GB for partition {par:s}. Capping memory at partition limit. "
            print(msg.format(lim=mem_lim, par=partition))
            mem_per_job = str(int(mem_lim)) + "GB"

    # Parse requested timeout period
    try:
        scalar_parser(timeout, varname="timeout", ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        raise exc

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
    _cluster_waiter(cluster, timeout)

    # Highlight how to connect to dask performance monitor
    print("Cluster dashboard accessible at {}".format(cluster.dashboard_link))

    return cluster


def _cluster_waiter(cluster, timeout):
    """
    Local helper that can be called recursively
    """

    # Wait until all workers have been started successfully or we run out of time
    wrkrs = cluster._count_active_workers()
    to = str(datetime.timedelta(seconds=timeout))[2:]
    total_workers = cluster._count_active_and_pending_workers()
    fmt = "{desc}: {n}/{total} \t[elapsed time {elapsed} | timeout at " + to + "]"
    ani = tqdm(desc="SLURM workers ready", total=total_workers,
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
    if counter == timeout:
        msg = "SLURM swarm could not be started within given time-out " +\
              "interval of {0:d} seconds"
        print(msg.format(timeout))
        query = "Do you want to [k]eep waiting for 60s, [a]bort or " +\
                "[c]ontinue with {0:d} workers?"
        choice = user_input(query.format(wrkrs), valid=["k", "a", "c"])

        if choice == "k":
            _cluster_waiter(cluster, 60)
        elif choice == "a":
            print("Closing cluster...")
            cluster.close()
            sys.exit()
        else:
            if wrkrs == 0:
                query = "Cannot continue with 0 workers. Do you want to " +\
                        "[k]eep waiting for 60s or [a]bort?"
                choice = user_input(query, valid=["k", "a"])
                if choice == "k":
                    _cluster_waiter(cluster, 60)
                else:
                    print("Closing cluster...")
                    cluster.close()
                    sys.exit()

    return
