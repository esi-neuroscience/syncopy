# -*- coding: utf-8 -*-
#
# Collection of I/O utility functions
#

# Builtin/3rd party package imports
import os
import sys
import shutil
import inspect
import numpy as np
from datetime import datetime
from glob import glob
from collections import OrderedDict
from tqdm import tqdm
if sys.platform == "win32":
    # tqdm breaks term colors on Windows - fix that (tqdm issue #446)
    import colorama
    colorama.deinit()
    colorama.init(strip=False)

# Local imports
from syncopy import __storage__, __sessionid__, __checksum_algorithm__, __spydir__
from syncopy.datatype.base_data import BaseData
from syncopy.datatype.util import get_dir_size
from syncopy.shared.parsers import scalar_parser
from syncopy.shared.errors import SPYTypeError, log
from syncopy.shared.queries import user_input

__all__ = ["cleanup", "clear"]

# Dictionary keys for beginning of info/json file that are not class properties
startInfoDict = OrderedDict()
startInfoDict["filename"] = None
startInfoDict["dataclass"] = None
startInfoDict["data_dtype"] = None
startInfoDict["data_shape"] = None
startInfoDict["data_offset"] = None
startInfoDict["trl_dtype"] = None
startInfoDict["trl_shape"] = None
startInfoDict["trl_offset"] = None
startInfoDict["file_checksum"] = None
startInfoDict["order"] = "C"
startInfoDict["checksum_algorithm"] = __checksum_algorithm__.__name__


def hash_file(fname, bsize=65536):
    """
    An enlightening docstring...

    Internal helper routine, do not parse inputs
    """

    hash = __checksum_algorithm__()
    with open(fname, "rb") as f:
        for block in iter(lambda: f.read(bsize), b""):
            hash.update(block)
    return hash.hexdigest()


def cleanup(older_than=24, interactive=True, only_current_session=False):
    """
    Delete old files in temporary Syncopy folder

    The location of the temporary folder is stored in `syncopy.__storage__`.

    Parameters
    ----------
    older_than : int
        Files older than `older_than` hours will be removed
    interactive : bool
        Set to `False` to remove all (sessions and dangling files) at once
        without a prompt asking for confirmation
    only_current_session : bool
        Set to `True` to only remove dangling files associated to *this*
        Syncopy instance

    Examples
    --------
    >>> spy.cleanup()
    """

    # Make sure age-cutoff is valid
    scalar_parser(older_than, varname="older_than", ntype="int_like",
                  lims=[0, np.inf])
    older_than = int(older_than)

    # For clarification: show location of storage folder that is scanned here
    funcName = "Syncopy <{}>".format(inspect.currentframe().f_code.co_name)
    storage_size_gb, storage_num_files = get_dir_size(__storage__, out="GB")
    dirInfo = \
        "\n{name:s} Analyzing temporary storage folder '{dir:s}' containing {numf:d} files with total size {sizegb:.2f} GB...\n"
    log(dirInfo.format(name=funcName, dir=__storage__, numf=storage_num_files, sizegb=storage_size_gb),
        caller='cleanup')

    # Parse interactive keyword: if `False`, don't ask, just delete
    if not isinstance(interactive, bool):
        raise SPYTypeError(interactive, varname="interactive", expected="bool")

    # Also check for dangling data (not associated to any session)
    data = glob(os.path.join(__storage__, "spy_*"))
    dangling = []
    for dat in data:
        sessid = os.path.splitext(os.path.basename(dat))[0].split("_")[1]
        if not only_current_session:
            dangling.append(dat)
        elif sessid == __sessionid__:
            dangling.append(dat)

    # Farewell if nothing's to do here
    if not dangling:
        ext = \
        "Did not find any dangling data or Syncopy session remains " +\
        "older than {age:d} hours."
        log(ext.format(name=funcName, age=older_than), caller=cleanup)
        spydir_size_gb, spydir_num_files = get_dir_size(__spydir__, out="GB")
        log(f"Note: {spydir_num_files} files with total size of {spydir_size_gb:.2f} GB left in spy dir '{__spydir__}'.",
            caller=cleanup)
        return

    # Prepare info prompt for dangling files
    if dangling:
        dangInfo = \
            "Found {numdang:d} dangling files not associated to any session " +\
            "using {szdang:4.1f} GB of disk space. \n"
        numdang = 0
        szdang = 0.0
        for file in dangling:
            try:
                if os.path.isfile(file):
                    szdang += os.path.getsize(file)/1024**3
                    numdang += 1
                elif os.path.isdir(file):
                    szdang += sum(os.path.getsize(os.path.join(dirpth, fname)) / 1024**3 \
                                           for dirpth, _, fnames in os.walk(file) \
                                               for fname in fnames)
                    numdang += 1

            except OSError as ex:
                log(f"Dangling file {file} no longer exists: {ex}. (Maybe already deleted.)", caller=cleanup)
        dangInfo = dangInfo.format(numdang=numdang, szdang=szdang)

        dangOptions = \
            "[D]ANGLING FILE removal to delete anything not associated to sessions " +\
            "(you will not be prompted for confirmation) \n"
        dangValid = ["D"]
        promptInfo = dangInfo
        promptOptions = dangOptions
        promptValid = dangValid

    # Put together actual prompt message message
    promptChoice = "\nPlease choose one of the following options:\n"
    abortOption = "[C]ANCEL\n"
    abortValid = ["C"]

    if dangling:
        rmAllOption = \
            "[R]EMOVE all dangling files at once " +\
            "(you will not be prompted for confirmation)\n"
        rmAllValid = ["R"]
        promptInfo = dangInfo
        promptOptions = dangOptions + rmAllOption
        promptValid = dangValid + rmAllValid

    # By default, ask what to do; if `interactive` is `False`, remove everything
    if interactive:
        choice = user_input(promptInfo + promptChoice + promptOptions + abortOption,
                            valid=promptValid + abortValid)
    else:
        choice = "R"

    # Deleate all dangling files at once
    if choice == "D":
        for dat in tqdm(dangling, desc="Deleting dangling data...", disable=None):
            _rm_session([dat])

    # Delete everything
    elif choice == "R":
        for contents in tqdm([[dat] for dat in dangling],
                             desc="Deleting temporary data...", disable=None):
            _rm_session(contents)

    # Don't do anything for now, continue w/dangling data
    else:
        print(f"Aborting...")

    # Report on remaining data
    storage_size_gb, storage_num_files = get_dir_size(__storage__, out="GB")
    log(f"{storage_num_files} files with total size of {storage_size_gb:.2f} GB left in storage dir '{__storage__}'.",
        caller='cleanup')
    spydir_size_gb, spydir_num_files = get_dir_size(__spydir__, out="GB")
    log(f"{spydir_num_files} files with total size of {spydir_size_gb:.2f} GB left in spy dir '{__spydir__}'.",
        caller='cleanup')


def clear():
    """
    Clear Syncopy objects from memory

    Notes
    -----
    Syncopy objects are **not** loaded wholesale into memory. Only the corresponding
    meta-information is read from disk and held in memory. The underlying numerical
    data is streamed on-demand from disk leveraging HDF5's modified LRU (least
    recently used) page replacement algorithm. Thus, :func:`syncopy.clear` simply
    force-flushes all of Syncopy's HDF5 backing devices to free up memory currently
    blocked by cached data chunks.

    Examples
    --------
    >>> spy.clear()
    """

    # Get current frame
    thisFrame = sys._getframe()

    # For later reference: dynamically fetch name of current function
    funcName = "Syncopy <{}>".format(thisFrame.f_code.co_name)

    # Go through caller's namespace and execute `clear` of `BaseData` children
    counter = 0
    for name, value in thisFrame.f_back.f_locals.items():
        if isinstance(value, BaseData):
            value.clear()
            counter += 1

    # Be talkative
    msg = "{name:s} flushed {objcount:d} objects from memory"
    print(msg.format(name=funcName, objcount=counter))

    return


def _rm_session(session_files):
    """
    Local helper for deleting tmp data of a given spy session
    """
    for file in session_files:
        try:
            os.unlink(file) if os.path.isfile(file) else shutil.rmtree(file)
        except Exception as ex:
            pass

    return
