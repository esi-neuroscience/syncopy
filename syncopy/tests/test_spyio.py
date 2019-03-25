# -*- coding: utf-8 -*-
#
# Test functionality of SyNCoPy-container I/O routines
# 
# Created: 2019-03-19 14:21:12
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-03-25 11:41:46>

import os
import tempfile
import pytest
import numpy as np
from syncopy.io import load_spy, hash_file
from syncopy.utils import SPYValueError, SPYTypeError
import syncopy.datatype as swd

class TestSpyIO(object):

    # Allocate test-datasets for AnalogData, SpectralData, SpikeData and EventData objects
    nc = 10
    ns = 30
    nt = 5
    nf = 15
    nd = 50
    data = {}
    trl = {}

    # Generate 2D array simulating an AnalogData array
    data["AnalogData"] = np.arange(1, nc*ns + 1).reshape(nc, ns)
    trl["AnalogData"] = np.vstack([np.arange(0, ns, 5),
                                   np.arange(5, ns + 5, 5),
                                   np.ones((int(ns/5), )),
                                   np.ones((int(ns/5), )) * np.pi]).T

    # Generate a 4D array simulating a SpectralData array
    data["SpectralData"] = np.arange(1, nc*ns*nt*nf + 1).reshape(ns, nt, nc, nf)
    trl["SpectralData"] = trl["AnalogData"]

    # Use a fixed random number generator seed to simulate a 2D SpikeData array
    seed = np.random.RandomState(13)
    data["SpikeData"] = np.vstack([seed.choice(ns, size=nd),
                                   seed.choice(nc, size=nd),
                                   seed.choice(int(nc/2), size=nd)]).T
    trl["SpikeData"] = trl["AnalogData"]

    # Define data classes to be used in tests below
    classes = ["AnalogData", "SpectralData", "SpikeData"]

    # Test correct handling of object log and cfg
    def test_logging(self):
        with tempfile.TemporaryDirectory() as tdir:
            for dclass in self.classes:
                dummy = getattr(swd, dclass)(self.data[dclass],
                                             trialdefinition=self.trl[dclass])
                ldum = len(dummy._log)
                dummy.save("dummy")

                # ensure saving is logged correctly
                assert len(dummy._log) > ldum
                assert dummy.cfg["method"] == "save_spy"

    # Test consistency of generated checksums
    def test_checksum(self):
        with tempfile.TemporaryDirectory() as tdir:
            for dclass in self.classes:
                fname = os.path.join(tdir, "dummy")
                
                dummy = getattr(swd, dclass)(self.data[dclass],
                                             trialdefinition=self.trl[dclass])
                dummy.save(fname)

                # perform checksum-matching - this must work
                dummy = load_spy(fname, checksum=True)

                # manipulate data file
                dat = np.array(dummy.data)
                dat += 1
                with open(dummy._filename, "wb") as fn:
                    np.save(fn, dat, allow_pickle=False)
                with pytest.raises(SPYValueError):
                    load_spy(fname, checksum=True)
                
                
    # def test_fname(self):
    #     asdf

    # CHECK FNAME
    # CHECK APPEND_EXT
    # CHECK MEMUSE
    # CHECK FILESIZE
    # CHEK STORAGE LOCATION (CORRECT FOLDER)
