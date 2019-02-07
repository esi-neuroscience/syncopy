# utils.py - Collection of I/O utility functions
# 
# Created: February  6 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-07 15:45:50>

# Builtin/3rd party package imports
import tempfile
from hashlib import blake2b

__all__ = ["FILE_EXT", "MANDATORY_ATTRS", "hash_file", "write_access"]

# Define SpykeWave's general file-/directory-naming conventions
FILE_EXT = {"dir" : ".spw",
            "json" : ".info",
            "data" : ".dat",
            "seg" : ".seg"}

# These attributes must be present in every valid SpykeWave BaseData object
MANDATORY_ATTRS = ["label", "segmentlabel", "log", "version"]

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
