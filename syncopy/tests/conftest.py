# -*- coding: utf-8 -*-
#
# pytest configuration that starts a SLURM cluster (if available)
#
# Created: 2019-07-05 15:22:24
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-05 15:50:08>

import pytest
from syncopy import __dask__


# If dask is available, either launch a SLURM cluster on a cluster node or 
# create a `LocalCluster` object if tests are run on a single machine. If dask
# is not installed, return a dummy None-valued cluster object (tests will be 
# skipped anyway)
if __dask__:
    import dask.distributed as dd
    from syncopy.shared import esi_cluster_setup
    from syncopy.tests.misc import is_slurm_node
    if is_slurm_node():
        cluster = esi_cluster_setup(partition="DEV", mem_per_job="4GB",
                                    timeout=600, interactive=False,
                                    start_client=False)
    else:
        cluster = dd.LocalCluster(n_workers=2)
else:
    cluster = None

@pytest.fixture
def testcluster():
    return cluster
