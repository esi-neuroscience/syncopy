# -*- coding: utf-8 -*-
#
# pytest configuration that starts a SLURM cluster (if available)
#
# Created: 2019-07-05 15:22:24
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-05 15:50:08>

import pytest
from syncopy.shared import esi_cluster_setup
from syncopy.tests.misc import is_slurm_node


# # Launch a SLURM cluster once and re-connect clients for all tests
# if is_slurm_node():
#     cluster = esi_cluster_setup(partition="DEV", mem_per_job="4GB",
#                                 timeout=600, interactive=False,
#                                 start_client=False)


# @pytest.fixture
# def esicluster():
#     return cluster
