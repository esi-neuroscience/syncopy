# -*- coding: utf-8 -*-
#
# Test functionality of SyNCoPy-container I/O routines
#
# Created: 2019-03-19 14:21:12
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-10 11:33:32>

import os
import tempfile
import shutil
import h5py
import pytest
import numpy as np
from numpy.lib.format import open_memmap
from glob import glob
from memory_profiler import memory_usage
from syncopy.datatype import AnalogData
from syncopy.datatype.base_data import VirtualData
from syncopy.io import save_spy, load_spy, FILE_EXT
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.tests.misc import is_slurm_node
import syncopy.datatype as swd

# Construct decorator for skipping certain tests
skip_in_slurm = pytest.mark.skipif(is_slurm_node(), reason="running on cluster node")


class TestSpyIO():

    # Allocate test-datasets for AnalogData, SpectralData, SpikeData and EventData objects
    nc = 10
    ns = 30
    nt = 5
    nf = 15
    nd = 50
    data = {}
    trl = {}

    # Generate 2D array simulating an AnalogData array
    data["AnalogData"] = np.arange(1, nc * ns + 1).reshape(ns, nc)
    trl["AnalogData"] = np.vstack([np.arange(0, ns, 5),
                                   np.arange(5, ns + 5, 5),
                                   np.ones((int(ns / 5), )),
                                   np.ones((int(ns / 5), )) * np.pi]).T

    # Generate a 4D array simulating a SpectralData array
    data["SpectralData"] = np.arange(1, nc * ns * nt * nf + 1).reshape(ns, nt, nc, nf)
    trl["SpectralData"] = trl["AnalogData"]

    # Use a fixed random number generator seed to simulate a 2D SpikeData array
    seed = np.random.RandomState(13)
    data["SpikeData"] = np.vstack([seed.choice(ns, size=nd),
                                   seed.choice(nc, size=nd),
                                   seed.choice(int(nc / 2), size=nd)]).T
    trl["SpikeData"] = trl["AnalogData"]

    # Generate bogus trigger timings
    data["EventData"] = np.vstack([np.arange(0, ns, 5),
                                   np.zeros((int(ns / 5), ))]).T
    data["EventData"][1::2, 1] = 1
    trl["EventData"] = trl["AnalogData"]

    # Define data classes to be used in tests below
    classes = ["AnalogData", "SpectralData", "SpikeData", "EventData"]

    # Test correct handling of object log and cfg
    def test_logging(self):
        with tempfile.TemporaryDirectory() as tdir:
            for dclass in self.classes:
                dname = os.path.join(tdir, "dummy")
                dummy = getattr(swd, dclass)(self.data[dclass],
                                             trialdefinition=self.trl[dclass],
                                             samplerate=1000)
                ldum = len(dummy._log)
                save_spy(dname, dummy)

                # ensure saving is logged correctly
                assert len(dummy._log) > ldum
                assert dummy.cfg["method"] == "save_spy"

            # Delete all open references to file objects b4 closing tmp dir
            del dummy

    # Test consistency of generated checksums
    @skip_in_slurm
    def test_checksum(self):
        with tempfile.TemporaryDirectory() as tdir:
            for dclass in self.classes:
                dname = os.path.join(tdir, "dummy")
                dummy = getattr(swd, dclass)(self.data[dclass],
                                             trialdefinition=self.trl[dclass],
                                             samplerate=1000)
                save_spy(dname, dummy)

                # perform checksum-matching - this must work
                dummy = load_spy(dname, checksum=True)

                # manipulate data file
                hname = dummy._filename
                del dummy
                h5f = h5py.File(hname, "r+")
                dset = h5f[dclass]
                dset[()] += 1
                h5f.close()
                with pytest.raises(SPYValueError):
                    load_spy(dname, checksum=True)
                shutil.rmtree(dname + ".spy")

    # Test correct handling of user-provided file-names
    def test_fname(self):
        with tempfile.TemporaryDirectory() as tdir:
            newname = "non_standard_name"
            for dclass in self.classes:
                dname = os.path.join(tdir, "dummy")
                dummy = getattr(swd, dclass)(self.data[dclass],
                                             trialdefinition=self.trl[dclass],
                                             samplerate=1000)
                save_spy(dname, dummy, fname=newname)

                # ensure provided file-name was actually used
                assert len(glob(os.path.join(dname + ".spy", newname + "*"))) == 2

                # load container using various versions of specific file-name
                fext = FILE_EXT.copy()
                dext = fext.pop("dir")
                flst = ["*" + fe for fe in fext.values()]
                for de in [dext, ""]:
                    for fe in flst + ["*", ""]:
                        dummy2 = load_spy(dname + de, fname=newname + fe)
                        for attr in ["data", "sampleinfo", "trialinfo"]:
                            assert np.array_equal(getattr(dummy, attr), getattr(dummy2, attr))

                # Delete all open references to file objects b4 closing tmp dir
                del dummy, dummy2
                shutil.rmtree(dname + dext)

    # Test if directory-name "extensions" work as intended
    def test_appendext(self):
        with tempfile.TemporaryDirectory() as tdir:
            for dclass in self.classes:
                dname = os.path.join(tdir, "dummy")
                dummy = getattr(swd, dclass)(self.data[dclass],
                                             trialdefinition=self.trl[dclass],
                                             samplerate=1000)
                save_spy(dname, dummy, append_extension=False)
                save_spy(dname, dummy, fname="preferred")

                # in case dir and dir.spw exist, prefence must be given to dir.spw
                dummy = load_spy(dname)
                assert "preferred" in dummy._filename

                # remove "regular" .spy-dir and re-load object from ".spy"-less dir
                del dummy
                shutil.rmtree(dname + ".spy")
                dummy = load_spy(dname)
                del dummy
                shutil.rmtree(dname)

    # Test memory usage when saving big VirtualData files
    def test_memuse(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "vdat.npy")
            dname = os.path.join(tdir, "dummy")
            vdata = np.ones((1000, 5000))  # ca. 38.2 MB
            np.save(fname, vdata)
            del vdata
            dmap = open_memmap(fname)
            adata = AnalogData(VirtualData([dmap, dmap, dmap]), samplerate=10)

            # Ensure memory consumption stays within provided bounds
            mem = memory_usage()[0]
            save_spy(dname, adata, memuse=60)
            assert (mem - memory_usage()[0]) < 70

            # Delete all open references to file objects b4 closing tmp dir
            del dmap, adata
