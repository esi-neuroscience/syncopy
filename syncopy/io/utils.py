# -*- coding: utf-8 -*-
#
# Collection of I/O utility functions
# 
# Created: 2019-02-06 14:30:17
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-06-07 14:22:11>

# Builtin/3rd party package imports
import os
import tempfile
import shutil
from datetime import datetime
from glob import glob
from hashlib import blake2b
from tqdm import tqdm
if sys.platform == "win32":
    # tqdm breaks term colors on Windows - fix that (tqdm issue #446)
    import colorama
    colorama.deinit()
    colorama.init(strip=False)

# Local imports
from syncopy import __storage__, __sessionid_
from syncopy.shared import scalar_parser_
from syncopy.shared.queries import user_yesno, user_input

# Define SynCoPy's general file-/directory-naming conventions
FILE_EXT = {"dir" : ".spy",
            "json" : ".info",
            "data" : ".dat"}

__all__ = ["FILE_EXT", "hash_file", "write_access", "cleanup"]


def hash_file(fname, bsize=65536):
    """
    An enlightening docstring...

    Internal helper routine, do not parse inputs
    """

    hash = blake2b()
    with open(fname, "rb") as f:
        for block in iter(lambda: f.read(bsize), b""):
            hash.update(block)
    return hash.hexdigest()    


def write_access(directory):
    """
    An enlightening docstring...

    Internal helper routine, do not parse inputs
    """

    try:
        with tempfile.TemporaryFile() as tmp:
            tmp.write(b"Alderaan shot first")
            tmp.seek(0)
            tmp.read()
        return True
    except:
        return False


def cleanup(older_than=24):
    """
    Docstring

    Remove dangling files older than `older_than` hrs
    """

    # Make sure age-cutoff is valid
    try:
        scalar_parser(older_than, varname="older_than", ntype="int_like",
                      lims=[0, np.inf])
    except Exception as exc:
        raise exc
    older_than = int(older_than)

    # Get current date + time and scan package's temp directory 
    now = datetime.now()
    sessions = glob(os.path.join(__storage__, "session*"))
    msg = "Found{numf:s} files created by session {sess:s} {age:d} " +\
          "days ago{sizeinfo:s}" +\
          " Do you want to permanently delete these files?"

    # Cycle through session-logs and identify stuff older than `older_than` hrs
    ses_list = []
    age_list = []
    usr_list = []
    siz_list = []
    for sess in sessions:
        sessid, sessowner, sessage = _get_session_id(sess)
        if sessid != __sessionid__ and sessage >= older_than:
                ses_list.append(sess)
                files = _get_session_files(sessid)
                age_list.append(round(sessage/24))
                usr_list.append(sessowner[:sessowner.find("@")])
                siz_list.append(_get_session_size(files, unit="GB"))

    # Abort if we didn't find any session data satisfying the provided criteria
    if len(ses_list) == 0:
        ext = "Did not find any syncopy session data older than {} hours. Exiting..."
        print(ext.format(older_than))
        return

    # Identify unique users
    usr_list = list(set(usr_list))
    
    # Ask the user how to proceed from here
    qst = "Found data of {numsess:d} syncopy sessions between {agemin:d} " +\
          "and {agemax:d} days old created by user{users:s} taking up a total " +\
          "of {gbsz:f4.1} GB of disk space. \n\n" +\
          "Do you want to\n" +\
          "[1] permanently delete all files at once (you will not be prompted for confirmation)?\n" +\
          "[2] go through each session and decide interactively?\n" +\
          "[3] abort?"
    choice = user_input(qst.format(numsess=len(ses_list),
                                   agemin=min(age_list),
                                   agemax=max(age_list),
                                   users="(s) " + ",".join(usr + ", " for usr in usr_list)[:-2] \
                                   if len(usr_list) > 1 else " " + usr_list[0],
                                   gbsz=sum(siz_list)),
                        valid=["1", "2", "3"])
    
    # Force-delete everything
    if choice == "1":
        for sess in tqdm(ses_list, desc="Deleting session data..."):
            _rm_session(sess)
            print("Done")

    # Query deletion for each session separately
    elif choice == "2":
        for sess in ses_list:
            sessid = _get_session_id(sess)
            files = _get_session_files(sessid)
            sz_fl = round(_get_session_size(files, unit="MB"))
            if user_yesno(msg.format(numf=" " + str(len(files)),
                                     sess=sesslog[:sesslog.find(":")],
                                     age=round(age_hr/24),
                                     sizeinfo=" using " + str(sz_fl) + \
                                     " MB of disk space." if len(files) else ".")):
                _rm_session(sess)

                    
def _rm_session(session):
    """
    Local helper for deleting tmp data of a given spy session
    """

    [os.unlink(file) if os.path.isfile(file) else shutil.rmtree(file) \
     for file in _get_session_files(_get_session_id(session))]
    os.unlink(session)

    return


def _get_session_info(session):
    """
    Local helper to extract session hash, log and age from session file

    This collection of one-liners is encapsulated in a separate function to 
    enforce a unique canonical way of extracting this infor from a session's
    log file
    """

    session_id = os.path.splitext(os.path.basename(session))[0].split("_")[1]
    with open(session, "r") as fid:
        session_log = fid.read()
    timestr = session_log[session_log.find("<") + 1:session_log.find(">")]
    timeobj = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')
    session_age = round((now - timeobj).total_seconds()/3600)
    session_owner = session_log[:session_log.find(":")]

    return session_id, session_owner, session_age


def _get_session_files(sessid):
    """
    Local helper that fetches all files found on disk related to provided session
    """

    return glob(os.path.join(__storage__, "*_{}_*".format(sessid)))


def _get_session_size(sessfiles, unit="MB"):
    """
    Local helper to calculate the on-disk size of provided session-related files
    """

    raw_size = sum(os.path.getsize(file) if os.path.isfile(file) else \
                   sum(os.path.getsize(os.path.join(dirpth, fname)) \
                       for dirpth, _, fnames in os.walk(file) \
                       for fname in fnames) for file in sessfiles)
    factor = 1024
    scale = {"GB": factor**3, "MB": factor**2, "KB": factor}

    return raw_size/scale[unit]
