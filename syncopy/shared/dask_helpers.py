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


def check_workers_available(client, timeout=120):
    """
    Checks for available (alive) Dask workers and waits max `timeout` seconds
    until at least a fraction of the workers is available.

    The minimum number of workers to be waited on depends
    on the total number of requested workers, and scales with:

        minWorkers = totalWorkers^0.7

    Meaning for 10 ``totalWorkers``, at least 5 have to be available,
    wheareas for 100 ``totalWorkers`` only 25 have to be available.
    """

    logger = get_logger()
    totalWorkers = len(client.cluster.requested)
    minWorkers = int(totalWorkers**0.7)

    # dictionary of workers
    workers = client.cluster.scheduler_info['workers']

    # some small initial wait
    sleep(.25)

    if len(workers) < minWorkers:
        logger.important(f"waiting for at least {minWorkers}/{totalWorkers} workers being available, timeout after {timeout} seconds..")
    client.wait_for_workers(minWorkers, timeout=timeout)

    # wait a little more to get consistent client print out
    sleep(.25)

    if len(workers) != totalWorkers:
        logger.important(f"{len(workers)}/{totalWorkers} workers available, starting computation..")

    # wait a little more to get consistent client print out
    sleep(.25)
