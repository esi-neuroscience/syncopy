# -*- coding: utf-8 -*-
# 
# 
# 
# Created: 2019-01-15 09:03:46
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2020-01-24 10:07:38>

# Builtin/3rd party package imports
import os
import sys
from hashlib import blake2b, sha1
import numpy as np

# Global version number
__version__ = "0.1a"

# Set up sensible printing options for NumPy arrays
np.set_printoptions(suppress=True, precision=4, linewidth=80)

# Set up dask configuration
try:
    import dask
    __dask__ = True
except ImportError:
    __dask__ = False
    print("\nSyncopy core: WARNING >> Could not import dask - Syncopy's parallel " +\
          "processing engine requires a working dask installation. " +\
          "Please consider installing dask and and dask_jobqueue using conda or pip. <<<")

# Define package-wide temp directory (and create it if not already present)
if os.environ.get("SPYTMPDIR"):
    __storage__ = os.path.abspath(os.path.expanduser(os.environ["SPYTMPDIR"]))
else:
    __storage__ = os.path.join(os.path.expanduser("~"), ".spy")
if not os.path.exists(__storage__):
    try:
        os.mkdir(__storage__)
    except Exception as exc:
        err = "Syncopy core: cannot create temporary storage directory {}. " +\
              "Original error message below\n{}"
        raise IOError(err.format( __storage__, str(exc)))

# Ensure Syncopy has write permissions and can actually use its temp storage    
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
__storagelimit__ = 10
with os.scandir(__storage__) as scan:
    st_fles = [fle.stat().st_size/1024**3 for fle in scan]
    st_size = sum(st_fles)
    if st_size > __storagelimit__:
        msg = "\nSyncopy core: WARNING >> Temporary storage folder {tmpdir:s} " +\
              "contains {nfs:d} files taking up a total of {sze:4.2f} GB on disk. \n" +\
              "Consider running `spy.cleanup()` to free up disk space. <<"
        print(msg.format(tmpdir=__storage__, nfs=len(st_fles), sze=st_size))

# Establish ID and log-file for current session
__sessionid__ = blake2b(digest_size=2, salt=os.urandom(blake2b.SALT_SIZE)).hexdigest()
__sessionfile__ = os.path.join(__storage__, "session_{}.id".format(__sessionid__))
        
# Set max. depth of traceback info shown in prompt
__tbcount__ = 5

# Set checksum algorithm to be used
__checksum_algorithm__ = sha1

# Fill up namespace
from . import shared, io, datatype, specest, statistics
from .shared import *
from .io import *
from .datatype import *
from .specest import *
from .statistics import *

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
