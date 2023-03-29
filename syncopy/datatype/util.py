"""
Helpers and tools for Syncopy data classes
"""

import os
from numbers import Number

# Syncopy imports
from syncopy import __storage__, __storagelimit__, __sessionid__
from syncopy.shared.errors import SPYTypeError, SPYValueError

__all__ = ['TrialIndexer']


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


def setup_storage():
    """
    Create temporary storage dir and report on its size.

    Returns
    -------
    storage_size: Size of files in temporary storage directory, in GB.
    storage_num_files: Number of files in temporary storage directory.
    """

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
        storage_size = 0.0
        storage_num_files = 0

        from pathlib import Path
        root_directory = Path('.')
        for f in root_directory.glob('**/*'):
            if f.is_file():
                try:
                    storage_size += f.stat().st_size / 1024 ** 3
                    storage_num_files += 1
                # this catches a cleanup by another process
                except FileNotFoundError:
                    continue

    return storage_size, storage_num_files



