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
from syncopy import __storage__, __sessionid__, __checksum_algorithm__
from syncopy.datatype.base_data import BaseData
from syncopy.shared.parsers import scalar_parser
from syncopy.shared.errors import SPYTypeError
from syncopy.shared.queries import user_yesno, user_input

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


def cleanup(older_than=24, interactive=True):
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
    dirInfo = \
        "\n{name:s} Analyzing temporary storage folder {dir:s}...\n"
    print(dirInfo.format(name=funcName, dir=__storage__))

    # Parse "hidden" interactive keyword: if `False`, don't ask, just delete
    if not isinstance(interactive, bool):
        raise SPYTypeError(interactive, varname="interactive", expected="bool")

    # Get current date + time and scan package's temp directory for session files
    now = datetime.now()
    sessions = glob(os.path.join(__storage__, "session*"))
    allIds = []
    for sess in sessions:
        allIds.append(os.path.splitext(os.path.basename(sess))[0].split("_")[1])

    # Also check for dangling data (not associated to any session)
    data = glob(os.path.join(__storage__, "spy_*"))
    dangling = []
    for dat in data:
        sessid = os.path.splitext(os.path.basename(dat))[0].split("_")[1]
        if sessid not in allIds:
            dangling.append(dat)

    # Cycle through session-logs and identify stuff older than `older_than` hrs
    sesList = []       # full path to session files
    ageList = []       # session age in days
    usrList = []       # session users
    sizList = []       # raw session sizes in bytes
    ownList = []       # session owners (user@machine)
    flsList = []       # files/directories associated to session
    for sk, sess in enumerate(sessions):
        sessid = allIds[sk]
        if sessid != __sessionid__:
            try:
               with open(sess, "r") as fid:
                    sesslog = fid.read()
                    timestr = sesslog[sesslog.find("<") + 1:sesslog.find(">")]
                    timeobj = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')
                    age = round((now - timeobj).total_seconds()/3600)   # age in hrs
                    if age >= older_than:
                        sesList.append(sess)
                        files = glob(os.path.join(__storage__, "*_{}_*".format(sessid)))
                        flsList.append(files)
                        ageList.append(round(age/24))                  # age in days
                        usrList.append(sesslog[:sesslog.find("@")])
                        ownList.append(sesslog[:sesslog.find(":")])
                        sizList.append(sum(os.path.getsize(file) if os.path.isfile(file) else sum(os.path.getsize(os.path.join(dirpth, fname)) \
                                       for dirpth, _, fnames in os.walk(file)
                                       for fname in fnames) for file in files))
            except OSError as ex:
                print(f"Unable to open {fid}: {ex}. (Maybe already deleted.)")

    # Farewell if nothing's to do here
    if not sesList and not dangling:
        ext = \
        "Did not find any dangling data or Syncopy session remains " +\
        "older than {age:d} hours."
        print(ext.format(name=funcName, age=older_than))
        return

    # Prepare session-related info prompt
    if sesList:
        usrList = list(set(usrList))
        gbList = [sz/1024**3 for sz in sizList]
        sesInfo = \
            "Found data of {numsess:d} syncopy sessions {ageinfo:s} " +\
            "created by user{users:s}'\ntaking up {gbinfo:s} of disk space. \n"
        sesInfo = sesInfo.format(numsess=len(sesList),
                                 ageinfo="between {agemin:d} and {agemax:d} days old".format(agemin=min(ageList),
                                                                                             agemax=max(ageList)) \
                                     if min(ageList) < max(ageList) else "from {} days ago".format(ageList[0]),
                                 users="(s) '" + ",".join(usr + ", " for usr in usrList)[:-2] \
                                     if len(usrList) > 1 else " '" + usrList[0],
                                 gbinfo="a total of {gbsz:4.1f} GB".format(gbsz=sum(gbList)) \
                                     if sum(gbList) > 1 else "less than 1 GB")
        sesOptions = \
            "[I]NTERACTIVE walkthrough to decide which session to remove \n" +\
            "[S]ESSION removal to delete all sessions at once " +\
            "(you will not be prompted for confirmation) \n"
        sesValid = ["I", "S"]
        promptInfo = sesInfo
        promptOptions = sesOptions
        promptValid = sesValid

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
                print(f"Dangling file {file} no longer exists: {ex}. (Maybe already deleted.)")
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

    if sesList and dangling:
        rmAllOption = \
            "[R]EMOVE all (sessions and dangling files) at once " +\
            "(you will not be prompted for confirmation)\n"
        rmAllValid = ["R"]
        promptInfo = sesInfo + dangInfo
        promptOptions = sesOptions + dangOptions + rmAllOption
        promptValid = sesValid + dangValid + rmAllValid

    # By default, ask what to do; if `interactive` is `False`, remove everything
    if interactive:
        choice = user_input(promptInfo + promptChoice + promptOptions + abortOption,
                            valid=promptValid + abortValid)
    else:
        choice = "R"

    # Query removal of data session by session
    if choice == "I":
        promptYesNo = \
            "Found{numf:s} files created by session {sess:s} {age:d} " +\
            "days ago{sizeinfo:s} Do you want to permanently delete these files?"
        for sk in range(len(sesList)):
            if user_yesno(promptYesNo.format(numf=" " + str(len(flsList[sk])),
                                             sess=ownList[sk],
                                             age=ageList[sk],
                                             sizeinfo=" using " + \
                                                 str(round(sizList[sk]/1024**2)) + \
                                                     " MB of disk space.")):
                _rm_session(flsList[sk])

    # Delete all session-remains at once
    elif choice == "S":
        for fls in tqdm(flsList, desc="Deleting session data..."):
            _rm_session(fls)

    # Deleate all dangling files at once
    elif choice == "D":
        for dat in tqdm(dangling, desc="Deleting dangling data..."):
            _rm_session([dat])

    # Delete everything
    elif choice == "R":
        for contents in tqdm(flsList + [[dat] for dat in dangling],
                        desc="Deleting temporary data..."):
            _rm_session(contents)

    # Don't do anything for now, continue w/dangling data
    else:
        print("Aborting...")

    return


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
