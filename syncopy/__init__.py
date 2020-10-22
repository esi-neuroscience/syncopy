# -*- coding: utf-8 -*-
# 
# Main package initializer
# 

# Builtin/3rd party package imports
import os
import sys
from hashlib import blake2b, sha1
import numpy as np

# Global version number
__version__ = "0.1b1"

# Set up sensible printing options for NumPy arrays
np.set_printoptions(suppress=True, precision=4, linewidth=80)

# Check dask configuration: (attempt to) import `dask` and `distributed`
try:
    import dask
    import dask.distributed as dd
    __dask__ = True
except ImportError:
    __dask__ = False
    msg = "\nSyncopy <core> WARNING: Could not import 'dask' and/or 'dask.distributed'. \n" +\
        "Syncopy's parallel processing engine requires a working dask installation. \n" +\
        "Please consider installing 'dask' as well as 'distributed' \n" +\
        "(and potentially 'dask_jobqueue'), e.g., via conda: \n" +\
        "\tconda install dask\n" +\
        "\tconda install distributed\n" +\
        "\tconda install -c conda-forge dask-jobqueue\n" +\
        "or using pip:\n" +\
        "\tpip install dask\n" +\
        "\tpip install distributed\n" +\
        "\tpip install dask-jobqueue"
    print(msg)

# Check if we're being imported by a parallel worker process
if __dask__:
    try: 
        dd.get_worker()
        __worker__ = True
    except ValueError:
        __worker__ = False
else:
    __worker__ = False

# (Try to) set up visualization environment
try:
    import matplotlib.pyplot as plt 
    import matplotlib.style as mplstyle
    import matplotlib as mpl
    __plt__ = True
except ImportError:
    __plt__ = False
    if not __worker__:
        msg = "\nSyncopy <core> WARNING: Could not import 'matplotlib'. \n" +\
            "Syncopy's plotting engine requires a working matplotlib installation. \n" +\
            "Please consider installing 'matplotlib', e.g., via conda: \n" +\
            "\tconda install matplotlib\n" +\
            "or using pip:\n" +\
            "\tpip install matplotlib"
        print(msg)

# Define package-wide temp directory (and create it if not already present)
if os.environ.get("SPYTMPDIR"):
    __storage__ = os.path.abspath(os.path.expanduser(os.environ["SPYTMPDIR"]))
else:
    __storage__ = os.path.join(os.path.expanduser("~"), ".spy")
if not __worker__ and not os.path.exists(__storage__):
    try:
        os.mkdir(__storage__)
    except Exception as exc:
        err = "Syncopy core: cannot create temporary storage directory {}. " +\
              "Original error message below\n{}"
        raise IOError(err.format( __storage__, str(exc)))

# Ensure Syncopy has write permissions and can actually use its temp storage    
if not __worker__:
    try:
        filePath = os.path.join(__storage__, "__test__.spy")
        with open(filePath, "ab+") as fileObj:
            fileObj.write(b"Alderaan shot first")
            fileObj.seek(0)
            fileObj.read()
        os.unlink(filePath)
    except Exception as exc:
        err = "Syncopy core: cannot access {}. Original error message below\n{}"
        raise IOError(err.format(__storage__, str(exc)))

# Check for upper bound of temp directory size (in GB)
if not __worker__:
    __storagelimit__ = 10
    with os.scandir(__storage__) as scan:
        st_fles = [fle.stat().st_size/1024**3 for fle in scan]
        st_size = sum(st_fles)
        if st_size > __storagelimit__:
            msg = "\nSyncopy <core> WARNING: Temporary storage folder {tmpdir:s} " +\
                "contains {nfs:d} files taking up a total of {sze:4.2f} GB on disk. \n" +\
                "Consider running `spy.cleanup()` to free up disk space."
            print(msg.format(tmpdir=__storage__, nfs=len(st_fles), sze=st_size))

# Establish ID and log-file for current session
__sessionid__ = blake2b(digest_size=2, salt=os.urandom(blake2b.SALT_SIZE)).hexdigest()
__sessionfile__ = os.path.join(__storage__, "session_{}.id".format(__sessionid__))
        
# Set max. depth of traceback info shown in prompt
if __worker__:
    __tbcount__ = 999
else:
    __tbcount__ = 5

# Set checksum algorithm to be used
__checksum_algorithm__ = sha1

# Fill up namespace
from . import shared, io, datatype, specest, statistics, plotting
from .shared import *
from .io import *
from .datatype import *
from .specest import *
from .statistics import *
from .plotting import *

# Register session
__session__ = datatype.base_data.SessionLogger()

# Override default traceback (differentiate b/w Jupyter/iPython and regular Python)
from .shared.errors import SPYExceptionHandler
try:
    ipy = get_ipython()
    import IPython
    IPython.core.interactiveshell.InteractiveShell.showtraceback = SPYExceptionHandler
    IPython.core.interactiveshell.InteractiveShell.showsyntaxerror = SPYExceptionHandler
    sys.excepthook = SPYExceptionHandler
except:
    sys.excepthook = SPYExceptionHandler

# Take care of `from syncopy import *` statements
__all__ = []
__all__.extend(datatype.__all__)
__all__.extend(io.__all__)
__all__.extend(shared.__all__)
__all__.extend(specest.__all__)
__all__.extend(statistics.__all__)
__all__.extend(plotting.__all__)
