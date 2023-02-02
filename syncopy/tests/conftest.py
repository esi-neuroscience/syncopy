# -*- coding: utf-8 -*-
#
# central pytest configuration
#

# Builtin/3rd party package imports
import sys
import pytest
from syncopy import __acme__
import syncopy.tests.test_packagesetup as setupTestModule
import dask.distributed as dd
import dask_jobqueue as dj
from syncopy.tests.misc import is_slurm_node

# If acme is available, either launch a SLURM cluster on a cluster node or
# create a `LocalCluster` object if tests are run on a single machine. If
# acme is not available, launch a custom SLURM cluster or again just a local
# cluster as fallback
cluster = None
if __acme__:
    from acme.dask_helpers import esi_cluster_setup
    if sys.platform != "win32":
        import resource
        if max(resource.getrlimit(resource.RLIMIT_NOFILE)) < 1024:
            msg = "Not enough open file descriptors allowed. Consider increasing " +\
                "the limit using, e.g., `ulimit -Sn 1024`"
            raise ValueError(msg)
    if is_slurm_node():
        cluster = esi_cluster_setup(partition="8GB", n_jobs=4,
                                    timeout=360, interactive=False,
                                    start_client=False)
    else:
        cluster = dd.LocalCluster(n_workers=4)
else:
    # manually start slurm cluster
    if is_slurm_node():
        n_jobs = 3
        reqMem = 32
        ESIQueue = 'S'
        slurm_wdir = "/cs/slurm/syncopy/"

        cluster = dj.SLURMCluster(cores=1, memory=f'{reqMem} GB', processes=1,
                                  local_directory=slurm_wdir,
                                  queue=f'{reqMem}GB{ESIQueue}',
                                  python=sys.executable)
        cluster.scale(n_jobs)
    else:
        cluster = dd.LocalCluster(n_workers=4)

# Set up a pytest fixture `testcluster` that uses the constructed cluster object
@pytest.fixture
def testcluster():
    return cluster

# Re-order tests to first run stuff in test_packagesetup.py, then everything else
def pytest_collection_modifyitems(items):

    # Collect tests to be run in this session and registered setup-related tests
    allTests = [testFunc.name if hasattr(testFunc, "name") else "" for testFunc in items]
    setupTests = [name for name in dir(setupTestModule)
                  if not name.startswith("__") and not name.startswith("@")]

    # If queried tests contain setup-tests, prioritize them
    newOrder = []
    for testFirst in setupTests:
        if testFirst in allTests:
            newOrder.append(allTests.index(testFirst))
    newOrder += [allTests.index(testFunc) for testFunc in allTests
                 if testFunc not in setupTests]

    # Save potentially re-ordered test sequence
    items[:] = [items[idx] for idx in newOrder]
