# -*- coding: utf-8 -*-
#
# central pytest configuration
#

# Builtin/3rd party package imports
import os
import importlib
import pytest
import syncopy
from syncopy import __acme__
import syncopy.tests.test_packagesetup as setupTestModule

# If dask is available, either launch a SLURM cluster on a cluster node or
# create a `LocalCluster` object if tests are run on a single machine. If dask
# is not installed, return a dummy None-valued cluster object (tests will be
# skipped anyway)
if __acme__:
    import dask.distributed as dd
    import resource
    from acme.dask_helpers import esi_cluster_setup
    from syncopy.tests.misc import is_slurm_node
    if max(resource.getrlimit(resource.RLIMIT_NOFILE)) < 1024:
        msg = "Not enough open file descriptors allowed. Consider increasing " +\
            "the limit using, e.g., `ulimit -Sn 1024`"
        raise ValueError(msg)
    if is_slurm_node():
        cluster = esi_cluster_setup(partition="8GBS", n_jobs=10,
                                    timeout=360, interactive=False,
                                    start_client=False)
    else:
        cluster = dd.LocalCluster(n_workers=2)
else:
    cluster = None

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
