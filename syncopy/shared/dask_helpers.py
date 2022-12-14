# -*- coding: utf-8 -*-
#
# Basic checkers to facilitate direct Dask interface
#

import subprocess
from time import sleep

# Syncopy import
from syncopy.shared.errors import SPYWarning, SPYInfo


def check_slurm_available():
    """
    Returns `True` if a SLURM instance could be reached via
    a `sinfo` call, `False` otherwise.
    """

    # Check if SLURM's `sinfo` can be accessed
    proc = subprocess.Popen("sinfo",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, shell=True)
    _, err = proc.communicate()
    # Any non-zero return-code means SLURM is not available
    # so we disable ACME
    if proc.returncode != 0:
        has_slurm = False
    else:
        has_slurm = True

    return has_slurm


def check_workers_available(cluster):
    """
    Tries to see the Dask workers and waits
    until all requested workers are available
    """

    totalWorkers = len(cluster.requested)    
    sec = 0
    workers = cluster.scheduler_info['workers']

    while len(workers) != totalWorkers:
        SPYInfo(f"{len(workers)}/{totalWorkers} workers available, waiting.. {sec}s")
        sleep(1)
        sec += 2
        workers = cluster.scheduler_info['workers']
    # wait a little more to get consistent client print out
    sleep(1)
