# utils.py - Collection of I/O utility functions
# 
# Created: February  6 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-06 16:58:54>

# Builtin/3rd party package imports
from hashlib import blake2b

__all__ = ["hash_file", "FILE_EXT"]

# Define SpykeWave's general file-/directory-naming conventions
FILE_EXT = {"out" : "spw",
            "json" : "info",
            "data" : "dat",
            "seg" : "seg"}

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
