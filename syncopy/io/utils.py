# -*- coding: utf-8 -*-
#
# Collection of I/O utility functions
# 
# Created: 2019-02-06 14:30:17
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-06-07 18:50:54>

# Builtin/3rd party package imports
import os
import sys
import tempfile
import shutil
import numpy as np
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
from syncopy import __storage__, __sessionid__
from syncopy.shared import scalar_parser
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
    all_ids = []
    for sess in sessions:
        all_ids.append(os.path.splitext(os.path.basename(sess))[0].split("_")[1])
    data = glob(os.path.join(__storage__, "spy_*"))
    dangling = []
    for dat in data:
        sessid = os.path.splitext(os.path.basename(dat))[0].split("_")[1]
        if sessid not in all_ids:
            dangling.append(dat)



    ## FIXME!!!: dangling objects lacking an associated session log file???
    # sessions += glob(os.path.join(__storage__, "spy_*"))
    
    # Cycle through session-logs and identify stuff older than `older_than` hrs
    ses_list = []       # full path to session files
    age_list = []       # session age in days
    usr_list = []       # session users
    siz_list = []       # raw session sizes in bytes
    sid_list = []       # session IDs (only the hashes)
    own_list = []       # session owners (user@machine)
    fls_list = []       # files/directories associated to session
    for sk, sess in enumerate(sessions):
        sessid = all_ids[sk]
        # sessid = os.path.splitext(os.path.basename(sess))[0].split("_")[1]
        if sessid != __sessionid__:
            with open(sess, "r") as fid:
                sesslog = fid.read()
            timestr = sesslog[sesslog.find("<") + 1:sesslog.find(">")]
            timeobj = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')
            age = round((now - timeobj).total_seconds()/3600)   # age in hrs
            if age >= older_than:
                ses_list.append(sess)
                files = glob(os.path.join(__storage__, "*_{}_*".format(sessid)))
                fls_list.append(files)
                age_list.append(round(age/24))                  # age in days
                usr_list.append(sesslog[:sesslog.find("@")])
                own_list.append(sesslog[:sesslog.find(":")])
                siz_list.append(sum(os.path.getsize(file) if os.path.isfile(file) else \
                                    sum(os.path.getsize(os.path.join(dirpth, fname)) \
                                        for dirpth, _, fnames in os.walk(file) \
                                        for fname in fnames) for file in files))

    # Abort if we didn't find any session data satisfying the provided criteria
    if len(ses_list) == 0:
        ext = "Syncopy cleanup: Did not find any syncopy session data older than {} hours."
        print(ext.format(older_than))

    else:

        # Format lists for output
        usr_list = list(set(usr_list))
        gb_list = [sz/1024**3 for sz in siz_list]

        # Ask the user how to proceed from here
        qst = "| Syncopy cleanup | Found data of {numsess:d} syncopy sessions between {agemin:d} " +\
              "and {agemax:d} days old created by user{users:s} taking up a total " +\
              "of {gbsz:4.1f} GB of disk space. \n\n" +\
              "Do you want to\n" +\
              "[1] permanently delete all files at once (you will not be prompted for confirmation)?\n" +\
              "[2] go through each session and decide interactively?\n" +\
              "[3] abort?\n"
        import ipdb; ipdb.set_trace()
        choice = user_input(qst.format(numsess=len(ses_list),
                                       agemin=min(age_list),
                                       agemax=max(age_list),
                                       users="(s) " + ",".join(usr + ", " for usr in usr_list)[:-2] \
                                       if len(usr_list) > 1 else " " + usr_list[0],
                                       gbsz=sum(gb_list)),
                            valid=["1", "2", "3"])

        # Force-delete everything
        if choice == "1":
            for sk, sess in enumerate(tqdm(ses_list, desc="Deleting session data...")):
                _rm_session(sess, fls_list[sk])

        # Query deletion for each session separately
        elif choice == "2":
            msg = "Found{numf:s} files created by session {sess:s} {age:d} " +\
                  "days ago{sizeinfo:s}" +\
                  " Do you want to permanently delete these files?"
            for sk, sess in enumerate(ses_list):
                if user_yesno(msg.format(numf=" " + str(len(fls_list[sk])),
                                         sess=own_list[sk],
                                         age=age_list[sk],
                                         sizeinfo=" using " + \
                                         str(round(siz_list[sk]/1024**2)) + \
                                         " MB of disk space.")):
                    _rm_session(sess, fls_list[sk])

        # Don't do anything, continue w/dangling data
        else:
            pass        

    # If we found data not associated to any registered session, ask what to do
    if len(dangling) > 0:
        qry = "| Syncopy cleanup | Found {numdang:d} dangling files not " +\
            "associated to any session using {szdang:4.1f} GB of disk space. \n\n" +\
            "Do you want to\n"
        
                
def _rm_session(session, session_files):
    """
    Local helper for deleting tmp data of a given spy session
    """

    [os.unlink(file) if os.path.isfile(file) else shutil.rmtree(file) \
     for file in session_files]

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
