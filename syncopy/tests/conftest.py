# -*- coding: utf-8 -*-
#
# pytest configuration that starts a parallel processing cluster (if available)
#
# Created: 2019-07-05 15:22:24
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-05 15:50:08>

import pytest
from syncopy import __dask__
import syncopy.tests.test_packagesetup as setupTestModule

# If dask is available, either launch a SLURM cluster on a cluster node or 
# create a `LocalCluster` object if tests are run on a single machine. If dask
# is not installed, return a dummy None-valued cluster object (tests will be 
# skipped anyway)
if __dask__:
    import dask.distributed as dd
    from syncopy.shared import esi_cluster_setup
    from syncopy.tests.misc import is_slurm_node
    if is_slurm_node():
        cluster = esi_cluster_setup(partition="DEV", n_jobs=10, mem_per_job="4GB",
                                    timeout=600, interactive=False,
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
