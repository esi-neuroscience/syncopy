# -*- coding: utf-8 -*-
#
# Main package initializer
#

# Builtin/3rd party package imports
import os
import sys
import subprocess
import getpass
import socket
import numpy as np
from hashlib import blake2b, sha1
from importlib.metadata import version, PackageNotFoundError
import dask.distributed as dd

# Get package version: either via meta-information from egg or via latest git commit
try:
    __version__ = version("esi-syncopy")
except PackageNotFoundError:
    proc = subprocess.Popen("git describe --tags",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, shell=True)
    out, err = proc.communicate()
    if proc.returncode != 0:
        proc = subprocess.Popen("git rev-parse HEAD:syncopy/__init__.py",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, shell=True)
        out, err = proc.communicate()
        if proc.returncode != 0:
            msg = "\nSyncopy <core> WARNING: Package is not installed in site-packages nor cloned via git. " +\
                "Please consider obtaining SyNCoPy sources from supported channels. "
            print(msg)
            out = "-999"
    __version__ = out.rstrip("\n")

# --- Greeting ---

def startup_print_once(message, force=False):
    """Print message once: do not spam message n times during all n worker imports.
    """
    try:
        dd.get_client()
    except ValueError:
        silence_file = os.path.join(os.path.expanduser("~"), ".spy", "silentstartup")
        if force or (os.getenv("SPYSILENTSTARTUP") is None and not os.path.isfile(silence_file)):
            print(message)


msg = f"""
Syncopy {__version__}

See https://syncopy.org for the online documentation.
For bug reports etc. please send an email to syncopy@esi-frankfurt.de
"""
startup_print_once(msg)

# Set up sensible printing options for NumPy arrays
np.set_printoptions(suppress=True, precision=4, linewidth=80)

# Check concurrent computing  setup (if acme is installed, dask is present too)
# Import `esi_cluster_setup` and `cluster_cleanup` from acme to make the routines
# available in the `spy` package namespace
try:
    from acme import esi_cluster_setup, cluster_cleanup
    __acme__ = True
except ImportError:
    __acme__ = False
    # ACME is critical on ESI infrastructure
    if socket.gethostname().startswith('esi-sv'):
        msg = "\nSyncopy <core> WARNING: Could not import Syncopy's parallel processing engine ACME. \n" +\
            "Please consider installing it via conda: \n" +\
            "\tconda install -c conda-forge esi-acme\n" +\
            "or using pip:\n" +\
            "\tpip install esi-acme"
        # do not spam via worker imports
        try:
            dd.get_client()
        except ValueError:
            print(msg)

# (Try to) set up visualization environment
try:
    import matplotlib.pyplot as plt
    import matplotlib.style as mplstyle
    import matplotlib as mpl
    __plt__ = True
except ImportError:
    __plt__ = False

# See if NWB is available
try:
    import pynwb
    __nwb__ = True
except ImportError:
    __nwb__ = False

# Set package-wide temp directory
csHome = "/cs/home/{}".format(getpass.getuser())
if os.environ.get("SPYDIR"):
    __spydir__ = os.path.abspath(os.path.expanduser(os.environ["SPYDIR"]))
    if not os.path.exists(__spydir__):
        raise ValueError(f"Environment variable SPYDIR set to non-existent or unreadable directory '{__spydir__}'. Please unset SPYDIR or create the directory.")
else:
    if os.path.exists(csHome): # ESI cluster.
        __spydir__ = os.path.join(csHome, ".spy")
    else:
        __spydir__ = os.path.abspath(os.path.join(os.path.expanduser("~"), ".spy"))

if os.environ.get("SPYTMPDIR"):
    __storage__ = os.path.abspath(os.path.expanduser(os.environ["SPYTMPDIR"]))
else:
    __storage__ = os.path.join(__spydir__, "tmp_storage")

if not os.path.exists(__spydir__):
        os.makedirs(__spydir__, exist_ok=True)

# Set upper bound for temp directory size (in GB)
__storagelimit__ = 10

# Establish ID and log-file for current session
__sessionid__ = blake2b(digest_size=2, salt=os.urandom(blake2b.SALT_SIZE)).hexdigest()

# Set max. no. of lines for traceback info shown in prompt
__tbcount__ = 5

# Set checksum algorithm to be used
__checksum_algorithm__ = sha1

# Fill namespace
from . import (
    shared,
    io,
    datatype)

from .shared import *
from .io import *
from .datatype import *
from .specest import *
from .connectivity import *
from .statistics import *
from .plotting import *
from .preproc import *

from .datatype.util import setup_storage, get_dir_size
storage_tmpdir_size_gb, storage_tmpdir_numfiles = setup_storage()  # Creates the storage dir if needed and computes size and number of files in there if any.
spydir_size_gb, spydir_numfiles = get_dir_size(__spydir__, out="GB")

from .shared.log import setup_logging
__logdir__ = None  # Gets set in setup_logging() call below.
setup_logging(spydir=__spydir__, session=__sessionid__)  # Sets __logdir__.
startup_print_once(f"Logging to log directory '{__logdir__}'.\nTemporary storage directory set to '{__storage__}'.\n")

storage_msg = (
        "\nSyncopy <core> WARNING: {folder_desc}:s '{tmpdir:s}' "
        + "contains {nfs:d} files taking up a total of {sze:4.2f} GB on disk. \n"
        + "Please run `spy.cleanup()` and/or manually free up disk space."
    )
if storage_tmpdir_size_gb > __storagelimit__:
    msg_formatted = storage_msg.format(folder_desc="Temporary storage folder", tmpdir=__storage__, nfs=storage_tmpdir_numfiles, sze=storage_tmpdir_size_gb)
    startup_print_once(msg_formatted, force=True)
else:
    # We also check the size of the whole Syncopy cfg folder, as older Syncopy versions placed files directly into it.
    if spydir_size_gb > __storagelimit__:
        msg_formatted = storage_msg.format(folder_desc="User config folder", tmpdir=__spydir__, nfs=spydir_numfiles, sze=spydir_size_gb)
        startup_print_once(msg_formatted, force=True)

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

from .shared.errors import log

# Manage user-exposed namespace imports
__all__ = []
__all__.extend(datatype.__all__)
__all__.extend(io.__all__)
__all__.extend(shared.__all__)
__all__.extend(specest.__all__)
__all__.extend(connectivity.__all__)
__all__.extend(statistics.__all__)
__all__.extend(plotting.__all__)
__all__.extend(preproc.__all__)
