# -*- coding: utf-8 -*-
#
# Test connectivity measures
#

# 3rd party imports
import matplotlib.pyplot as ppl

# Local imports
from syncopy import __acme__
if __acme__:
    import dask.distributed as dd
from syncopy.connectivity import connectivity
