# -*- coding: utf-8 -*-
#
# Collection of I/O utility functions
# 
# Created: 2019-02-06 14:30:17
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-04-15 13:42:15>

# Builtin/3rd party package imports
import os
import tempfile
from datetime import datetime
from glob import glob
from hashlib import blake2b

# Local imports
from syncopy import __storage__, __sessionid__

__all__ = ["FILE_EXT", "hash_file", "write_access", "user_yesno", "cleanup"]

# Define SynCoPy's general file-/directory-naming conventions
FILE_EXT = {"dir" : ".spy",
            "json" : ".info",
            "data" : ".dat",
            "trl" : ".trl"}


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


def user_yesno(msg, default=None):
    """
    Docstring
    """

    # Parse optional `default` answer
    valid = {"yes": True, "y": True, "ye":True, "no":False, "n":False}
    if default is None:
        suffix = " [y/n] "
    elif default == "yes":
        suffix = " [Y/n] "
    elif default == "no":
        suffix = " [y/N] "

    # Start actually doing something
    while True:
        choice = input(msg + suffix).lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid.keys():
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def cleanup():
    """
    Docstring

    Remove dangling files older than 24hrs
    """

    # Get current date + time and scan package's temp directory 
    now = datetime.now()
    sessions = glob(os.path.join(__storage__, "session*"))
    msg = "Found {numf:d} files created by session {sess:s} {age:3.1f} days ago." +\
          " Do you want to permanently delete these files?"

    # Cycle through session-logs and identify those older than 24hrs
    for sess in sessions:
        sessid = os.path.splitext(os.path.basename(sess))[0].split("_")[1]
        if sessid != __sessionid__:
            with open(sess, "r") as fid:
                sesslog = fid.read()
            timestr = sesslog[sesslog.find("<") + 1:sesslog.find(">")]
            timeobj = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')
            age_hr = (now - timeobj).total_seconds()/3600
            if age_hr >= 24:
                files = glob(os.path.join(__storage__, "*_{}_*".format(sessid)))
                sz_fl = sum([os.stat(file).st_size/1024**2 for file in files])
                if user_yesno(msg.format(numf=len(files),
                                         sess=sesslog[:sesslog.find(":")],
                                         age=age_hr/24)):
                    [os.unlink(file) for file in files]
                    os.unlink(sess)
