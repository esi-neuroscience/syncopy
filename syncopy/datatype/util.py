"""
Helpers and tools for Syncopy data classes
"""

import os
import getpass
import socket
from datetime import datetime
from numbers import Number

# Syncopy imports
from syncopy import __storage__, __storagelimit__, __sessionid__
from syncopy.shared.errors import SPYTypeError, SPYValueError

__all__ = ['TrialIndexer', 'SessionLogger']


class TrialIndexer:

    def __init__(self, data_object, idx_list):
        """
        Class to obtain an indexable trials iterable from
        an instantiated Syncopy data class `data_object`.
        Relies on the `_get_trial` method of the
        respective `data_object`.

        Parameters
        ----------
        data_object : Syncopy data class, e.g. AnalogData

        idx_list : list
            List of valid trial indices for `_get_trial`
        """

        self.data_object = data_object
        self.idx_list = idx_list
        self._len = len(idx_list)

    def __getitem__(self, trialno):
        # single trial access via index operator []
        if not isinstance(trialno, Number):
            raise SPYTypeError(trialno, "trial index", "single number to index a single trial")
        if trialno not in self.idx_list:
            lgl = "index of existing trials"
            raise SPYValueError(lgl, "trial index", trialno)
        return self.data_object._get_trial(trialno)

    def __iter__(self):
        # this generator gets freshly created and exhausted
        # for each new iteration, with only 1 trial being in memory
        # at any given time
        yield from (self[i] for i in self.idx_list)

    def __len__(self):
        return self._len

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "{} element iterable".format(self._len)


class SessionLogger:

    __slots__ = ["sessionfile", "_rm"]

    def __init__(self):

        # Create package-wide tmp directory if not already present
        if not os.path.exists(__storage__):
            try:
                os.mkdir(__storage__)
            except Exception as exc:
                err = (
                    "Syncopy core: cannot create temporary storage directory {}. "
                    + "Original error message below\n{}"
                )
                raise IOError(err.format(__storage__, str(exc)))

        # Check for upper bound of temp directory size
        with os.scandir(__storage__) as scan:
            st_size = 0.0
            st_fles = 0
            for fle in scan:
                try:
                    st_size += fle.stat().st_size / 1024 ** 3
                    st_fles += 1
                # this catches a cleanup by another process
                except FileNotFoundError:
                    continue

            if st_size > __storagelimit__:
                msg = (
                    "\nSyncopy <core> WARNING: Temporary storage folder {tmpdir:s} "
                    + "contains {nfs:d} files taking up a total of {sze:4.2f} GB on disk. \n"
                    + "Consider running `spy.cleanup()` to free up disk space."
                )
                print(msg.format(tmpdir=__storage__, nfs=st_fles, sze=st_size))

        # If we made it to this point, (attempt to) write the session file
        sess_log = "{user:s}@{host:s}: <{time:s}> started session {sess:s}"
        self.sessionfile = os.path.join(
            __storage__, "session_{}_log.id".format(__sessionid__)
        )
        try:
            with open(self.sessionfile, "w") as fid:
                fid.write(
                    sess_log.format(
                        user=getpass.getuser(),
                        host=socket.gethostname(),
                        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        sess=__sessionid__,
                    )
                )
        except Exception as exc:
            err = "Syncopy core: cannot access {}. Original error message below\n{}"
            raise IOError(err.format(self.sessionfile, str(exc)))

        # Workaround to prevent Python from garbage-collecting ``os.unlink``
        self._rm = os.unlink

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Session {}".format(__sessionid__)

    def __del__(self):
        try:
            self._rm(self.sessionfile)
        except FileNotFoundError:
            pass
    
