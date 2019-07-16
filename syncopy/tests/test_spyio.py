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
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYError
import syncopy.datatype as swd
from syncopy.tests.misc import generate_artifical_data

class TestSpyIO():

    # Test correct handling of object log and cfg
    def test_logging(self):
        with tempfile.TemporaryDirectory() as tdir:            
            dummy = generate_artifical_data(inmemory=True)
            ldum = len(dummy._log)
            save_spy(dummy, filename=os.path.join(tdir, "dummy"))

            # ensure saving is logged correctly
            assert len(dummy._log) > ldum
            assert dummy.cfg["method"] == "save_spy"
            
            # Delete all open references to file objects b4 closing tmp dir
            del dummy


    # Test correct handling of user-provided file-names
    def test_save_fname(self):
        with tempfile.TemporaryDirectory() as tdir:
            dummy = generate_artifical_data(inmemory=True)
            
            # filename without extension
            filename = "some_filename"
            save_spy(dummy, filename=os.path.join(tdir, filename))
            assert len(glob(os.path.join(tdir, filename + "*"))) == 2            
            
            # filename with extension
            filename = "some_filename_w_ext.analog"
            save_spy(dummy, filename=os.path.join(tdir, filename))
            assert len(glob(os.path.join(tdir, filename + "*"))) == 2
            
            # container with extension
            container = "test_container.spy"
            save_spy(dummy, container=os.path.join(tdir, container))
            assert len(glob(os.path.join(tdir, container, "*"))) == 2
            
            # container w/o extension
            container = "test_container2"
            save_spy(dummy, container=os.path.join(tdir, container))
            assert len(glob(os.path.join(tdir, container + ".spy", "*"))) == 2
            
            # container with extension and tag
            container = "test_container.spy"
            tag = "sometag"
            save_spy(dummy, container=os.path.join(tdir, container), tag=tag)
            assert len(glob(os.path.join(tdir, container, "test_container_sometag*"))) == 2
            
            # both container and filename
            with pytest.raises(SPYError):
                save_spy(dummy, container="container", filename="someFile")
            
            # neither container nor filename
            with pytest.raises(SPYError):
                save_spy(dummy)
            
            del dummy                

  
    def test_save_mmap(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "vdat.npy")
            dname = os.path.join(tdir, "dummy")
            vdata = np.ones((1000, 5000))  # ca. 38.2 MB
            np.save(fname, vdata)
            del vdata
            dmap = open_memmap(fname)
            adata = AnalogData(dmap, samplerate=10)

            # Ensure memory consumption stays within provided bounds
            mem = memory_usage()[0]
            save_spy(adata, filename=dname, memuse=60)
            assert (mem - memory_usage()[0]) < 70

            # Delete all open references to file objects b4 closing tmp dir
            del dmap, adata
