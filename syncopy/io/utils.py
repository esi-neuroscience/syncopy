# -*- coding: utf-8 -*-
#
# Collection of I/O utility functions
# 
# Created: 2019-02-06 14:30:17
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-03-04 15:06:18>

# Builtin/3rd party package imports
import tempfile
from hashlib import blake2b

__all__ = ["FILE_EXT", "hash_file", "write_access"]

# Define SynCoPy's general file-/directory-naming conventions
FILE_EXT = {"dir" : ".spy",
            "json" : ".info",
            "data" : ".dat",
            "trl" : ".trl"}

##########################################################################################
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

##########################################################################################
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
