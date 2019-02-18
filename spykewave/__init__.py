# __init__.py - Import SpykeWave sub-packages
# 
# Created: January 15 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-18 17:39:55>

# Global version number
__version__ = "0.1a"

# Set up sensible printing options for NumPy arrays
import numpy as np
np.set_printoptions(suppress=True, precision=4, linewidth=80)

# Set up dask configuration
import socket
try:
    from dask.distributed import Client, LocalCluster, get_client
    __dask__ = True
except ImportError:
    __dask__ = False
if __dask__:
    try:
        client = get_client()
    except:
        cluster = LocalCluster(processes=False)
        client = Client(cluster)

# Fill up namespace
from .utils import *
from .io import *
from .datatype import *
from .specest import *

# Take care of `from spykewave import *` statements
__all__ = []
__all__.extend(datatype.__all__)
__all__.extend(io.__all__)
__all__.extend(utils.__all__)
__all__.extend(specest.__all__)
__all__.extend([__version__, __dask__])
