# -*- coding: utf-8 -*-
#
# Basic checkers to facilitate direct Dask interface
#

import subprocess
from time import sleep

# Syncopy imports
from syncopy.shared.errors import SPYWarning, SPYInfo
from .log import get_logger


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


def check_workers_available(client, n_workers=1, timeout=120):
    """
    Checks for available (alive) Dask workers and waits max `timeout` seconds
    until at least ``n_workers`` workers are available.
    """

    logger = get_logger()
    totalWorkers = len(client.cluster.requested)

    # dictionary of workers
    workers = client.cluster.scheduler_info['workers']

    # some small initial wait
    sleep(.25)

    if len(workers) < n_workers:
        logger.important(f"waiting for at least {n_workers}/{totalWorkers} workers being available, timeout after {timeout} seconds..")
    client.wait_for_workers(n_workers, timeout=timeout)

    sleep(.25)

    # report what we have
    logger.important(f"{len(workers)}/{totalWorkers} workers available, starting computation..")

    # wait a little more to get consistent client print out
    sleep(.25)
