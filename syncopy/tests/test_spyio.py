# -*- coding: utf-8 -*-
#
# Test functionality of SyNCoPy-container I/O routines
#

# Builtin/3rd party package imports
import os
import tempfile
import shutil
import h5py
import time
import pytest
import numpy as np
from numpy.lib.format import open_memmap
from glob import glob
from memory_profiler import memory_usage

# Local imports
from syncopy.datatype import AnalogData
from syncopy.io import save, load
from syncopy.shared.filetypes import FILE_EXT
from syncopy.shared.errors import SPYValueError, SPYIOError, SPYError
import syncopy.datatype as swd
from syncopy.tests.misc import generate_artificial_data

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
    data["AnalogData"] = np.arange(1, nc*ns + 1).reshape(ns, nc)
    trl["AnalogData"] = np.vstack([np.arange(0, ns, 5),
                                   np.arange(5, ns + 5, 5),
                                   np.ones((int(ns/5), )),
                                   np.ones((int(ns/5), )) * np.pi]).T

    # Generate a 4D array simulating a SpectralData array
    data["SpectralData"] = np.arange(1, nc*ns*nt*nf + 1).reshape(ns, nt, nc, nf)
    trl["SpectralData"] = trl["AnalogData"]

    # Generate a 4D array simulating a CorssSpectralData array
    data["CrossSpectralData"] = np.arange(1, nc*nc*ns*nf + 1).reshape(ns, nf, nc, nc)
    trl["CrossSpectralData"] = trl["AnalogData"]

    # Use a fixed random number generator seed to simulate a 2D SpikeData array
    seed = np.random.RandomState(13)
    data["SpikeData"] = np.vstack([seed.choice(ns, size=nd),
                                   seed.choice(nc, size=nd),
                                   seed.choice(int(nc/2), size=nd)]).T
    trl["SpikeData"] = trl["AnalogData"]

    # Generate bogus trigger timings
    data["EventData"] = np.vstack([np.arange(0, ns, 5),
                                   np.zeros((int(ns/5), ))]).T
    data["EventData"][1::2, 1] = 1
    trl["EventData"] = trl["AnalogData"]

    # Define data classes to be used in tests below
    classes = ["AnalogData", "SpectralData", "CrossSpectralData", "SpikeData", "EventData"]

    # Test correct handling of object log and cfg
    def test_logging(self):
        with tempfile.TemporaryDirectory() as tdir:
            fname = os.path.join(tdir, "dummy")
            dummy = generate_artificial_data(inmemory=True)
            ldum = len(dummy._log)
            save(dummy, filename=fname)

            # ensure saving is logged correctly
            assert len(dummy._log) > ldum
            assert dummy.filename in dummy._log
            assert dummy.filename + FILE_EXT["info"] in dummy._log
            assert dummy.cfg["method"] == "save"
            assert dummy.filename in dummy.cfg["files"]
            assert dummy.filename + FILE_EXT["info"] in dummy.cfg["files"]

            # ensure loading is logged correctly
            dummy2 = load(filename=fname + ".analog")
            assert len(dummy2._log) > len(dummy._log)
            assert dummy2.filename in dummy2._log
            assert dummy2.filename + FILE_EXT["info"] in dummy._log
            assert dummy2.cfg.cfg["method"] == "load"
            assert dummy2.filename in dummy2.cfg.cfg["files"]
            assert dummy2.filename + FILE_EXT["info"] in dummy2.cfg.cfg["files"]

            # Delete all open references to file objects b4 closing tmp dir
            del dummy, dummy2

    # Test consistency of generated checksums
    def test_checksum(self):
        with tempfile.TemporaryDirectory() as tdir:
            for dclass in self.classes:
                dname = os.path.join(tdir, "dummy")
                dummy = getattr(swd, dclass)(self.data[dclass], samplerate=1000)
                dummy.trialdefintion = self.trl[dclass]
                save(dummy, dname)

                # perform checksum-matching - this must work
                dummy2 = load(dname, checksum=True)

                # manipulate data file
                hname = dummy._filename
                del dummy, dummy2
                time.sleep(0.1)  # wait to kick-off garbage collection
                h5f = h5py.File(hname, "r+")
                dset = h5f["data"]
                # provoke checksum error by adding 1 to all datasets
                dset[()] += 1
                h5f.close()

                with pytest.raises(SPYValueError):
                    load(dname, checksum=True)
                shutil.rmtree(dname + ".spy")

    # Test correct handling of user-provided file-names
    def test_save_fname(self):
        for dclass in self.classes:
            with tempfile.TemporaryDirectory() as tdir:
                dummy = getattr(swd, dclass)(self.data[dclass], samplerate=1000)
                dummy.trialdefintion = self.trl[dclass]

                # object w/o container association
                with pytest.raises(SPYError):
                    dummy.save()

                # filename without extension
                filename = "some_filename"
                save(dummy, filename=os.path.join(tdir, filename))
                assert len(glob(os.path.join(tdir, filename + "*"))) == 2

                # filename with extension
                filename = "some_filename_w_ext" + dummy._classname_to_extension()
                save(dummy, filename=os.path.join(tdir, filename))
                assert len(glob(os.path.join(tdir, filename + "*"))) == 2

                # filename with invalid extension
                filename = "some_filename_w_ext.invalid"
                with pytest.raises(SPYError):
                    save(dummy, filename=os.path.join(tdir, filename))

                # filename with multiple extensions
                filename = "some_filename.w.ext" + dummy._classname_to_extension()
                with pytest.raises(SPYError):
                    save(dummy, filename=os.path.join(tdir, filename))

                # container with extension
                container = "test_container.spy"
                save(dummy, container=os.path.join(tdir, container))
                assert len(glob(os.path.join(tdir, container, "*"))) == 2

                # container with invalid extension
                container = "test_container.invalid"
                with pytest.raises(SPYError):
                    save(dummy, container=os.path.join(tdir, container))

                # container with multiple extensions
                container = "test_container.invalid.too"
                with pytest.raises(SPYValueError):
                    save(dummy, container=os.path.join(tdir, container))

                # container w/o extension
                container = "test_container2"
                save(dummy, container=os.path.join(tdir, container))
                assert len(glob(os.path.join(tdir, container + ".spy", "*"))) == 2

                # container with extension and tag
                container = "test_container.spy"
                tag = "sometag"
                save(dummy, container=os.path.join(tdir, container), tag=tag)
                assert len(glob(os.path.join(tdir, container, "test_container_sometag*"))) == 2

                # explicit overwrite
                save(dummy, container=os.path.join(tdir, container), tag=tag, overwrite=True)
                assert len(glob(os.path.join(tdir, container, "test_container_sometag*"))) == 2

                # implicit overwrite
                dummy.save()
                assert len(glob(os.path.join(tdir, container, "test_container_sometag*"))) == 2

                # attempted overwrite w/o keyword
                with pytest.raises(SPYIOError):
                    save(dummy, container=os.path.join(tdir, container), tag=tag)

                # shortcut with new tag
                dummy.save(tag="newtag")
                assert len(glob(os.path.join(tdir, container, "test_container_newtag*"))) == 2

                # overwrite new tag
                dummy.save(tag="newtag", overwrite=True)
                assert len(glob(os.path.join(tdir, container, "test_container_newtag*"))) == 2

                # attempted overwrite w/o keyword
                with pytest.raises(SPYIOError):
                    dummy.save(tag="newtag")

                # both container and filename
                with pytest.raises(SPYError):
                    save(dummy, container="container", filename="someFile")

                # neither container nor filename
                with pytest.raises(SPYError):
                    save(dummy)

                del dummy

    # Test saving/loading single files from containers
    def test_container_saveload(self):
        with tempfile.TemporaryDirectory() as tdir:
            for dclass in self.classes:
                dummy = getattr(swd, dclass)(self.data[dclass], samplerate=1000)
                dummy.trialdefintion = self.trl[dclass]

                # load single file from container
                container = "single_container_" + dummy._classname_to_extension()[1:]
                dummy.save(container=os.path.join(tdir, container))
                dummy2 = load(os.path.join(tdir, container))
                for attr in ["data", "sampleinfo", "trialinfo"]:
                    assert np.array_equal(getattr(dummy, attr), getattr(dummy2, attr))
                del dummy2

                # load single file via dataclass
                dummy2 = load(os.path.join(tdir, container),
                              dataclass=dummy._classname_to_extension())

                # save and load single file via tag
                container2 = "another_single_container_" + dummy._classname_to_extension()[1:]
                dummy2.save(container=os.path.join(tdir, container2), tag="sometag")
                dummy3 = load(os.path.join(tdir, container2), tag="sometag")
                for attr in ["data", "sampleinfo", "trialinfo"]:
                    assert np.array_equal(getattr(dummy2, attr), getattr(dummy3, attr))
                del dummy, dummy2, dummy3

                # tag mismatch in single-file container
                with pytest.raises(SPYIOError):
                    load(os.path.join(tdir, container2), tag="invalid")

                # dataclass mismatch in single-file container
                wrong_ext = getattr(swd, list(set(self.classes).difference([dclass]))[0])()._classname_to_extension()
                with pytest.raises(SPYIOError):
                    load(os.path.join(tdir, container), dataclass=wrong_ext)

                # invalid dataclass specification
                with pytest.raises(SPYValueError):
                    load(os.path.join(tdir, container), dataclass='.invalid')

    # Test saving multiple objects to the same container
    def test_multi_saveload(self):
        with tempfile.TemporaryDirectory() as tdir:
            for dk, dclass in enumerate(self.classes):
                dummy = getattr(swd, dclass)(self.data[dclass], samplerate=1000)
                dummy.trialdefintion = self.trl[dclass]

                # save to joint container
                container = "multi_container"
                dummy.save(container=os.path.join(tdir, container))

                # try to load last class from joint container before it's there
                if dk == len(self.classes) - 2:
                    not_yet = getattr(swd, self.classes[dk + 1])()._classname_to_extension()
                    with pytest.raises(SPYIOError):
                        load(os.path.join(tdir, container), dataclass=not_yet)

                # try to load everything but the first class at the beginning
                if dk == 0:
                    ext_list = []
                    for attr in self.classes[1:]:
                        ext_list.append(getattr(swd, attr)()._classname_to_extension())
                    fname = os.path.join(os.path.join(tdir, container + FILE_EXT["dir"]),
                                         container + dummy._classname_to_extension())
                    with pytest.raises(SPYValueError):
                        load(fname, dataclass=ext_list)

                del dummy

            # load all files created above
            container = "multi_container"
            objDict = load(os.path.join(tdir, container))
            assert len(objDict.keys()) == len(self.classes)
            all_ext = []
            for attr in self.classes:
                all_ext.append(getattr(swd, attr)()._classname_to_extension())
            fnameList = []
            for ext in all_ext:
                fnameList.append(os.path.join(os.path.join(tdir, container + FILE_EXT["dir"]),
                                              container + ext))
            for name, obj in objDict.items():
                assert obj.filename in fnameList
                fname = fnameList.pop(fnameList.index(obj.filename))
                assert name in fname
            assert len(fnameList) == 0
            del objDict

            # load single file from joint container via dataclass
            dummy = load(os.path.join(tdir, container), dataclass="analog")
            assert dummy.filename == os.path.join(os.path.join(tdir, container + FILE_EXT["dir"]),
                                                  container + ".analog")
            dummy.save(tag="2ndanalog")
            del dummy

            # load single file from joint container using tag
            dummy = load(os.path.join(tdir, container), tag="2ndanalog")
            dummy.save(tag="3rdanalog")
            del dummy

            # load single file from joint container using dataclass and tag
            dummy = load(os.path.join(tdir, container), dataclass="analog", tag="3rdanalog")

            # load single file from joint container using multiple dataclasses and single tag
            dummy2 = load(os.path.join(tdir, container), dataclass=["analog", "spectral"],
                          tag="3rdanalog")
            assert dummy2.filename == dummy.filename

            # load single file from joint container using single dataclass and multiple tags
            dummy3 = load(os.path.join(tdir, container), dataclass="analog",
                          tag=["3rdanalog", "invalid"])
            assert dummy3.filename == dummy.filename

            # load single file from joint container using multiple dataclasses and tags
            dummy4 = load(os.path.join(tdir, container), dataclass=["analog", "spectral"],
                          tag=["3rdanalog", "invalid"])
            assert dummy4.filename == dummy.filename
            del dummy, dummy2, dummy3, dummy4

            # load multiple files from joint container using single tag
            objDict = load(os.path.join(tdir, container), tag="analog")
            assert len(objDict.keys()) == 2
            wanted = ["2nd", "3rd"]
            for name in objDict.keys():
                inWanted = [tag in name for tag in wanted]
                assert any(inWanted)
                inWanted.pop(inWanted.index(True))
            del objDict

            # load multiple files from joint container using multiple tags
            objDict = load(os.path.join(tdir, container), tag=["2nd", "3rd"])
            assert len(objDict.keys()) == 2
            wanted = ["2nd", "3rd"]
            for name in objDict.keys():
                inWanted = [tag in name for tag in wanted]
                assert any(inWanted)
                inWanted.pop(inWanted.index(True))
            del objDict

            # load all AnalogData files from joint container via single dataclass
            objDict = load(os.path.join(tdir, container), dataclass="analog")
            assert len(objDict.keys()) == 3
            wanted = ["3rdanalog", "2ndanalog", "analog"]
            for name in objDict.keys():
                inWanted = [tag in name for tag in wanted]
                assert any(inWanted)
                inWanted.pop(inWanted.index(True))
            del objDict

            # load multiple files from joint container via multiple dataclasses
            wanted = ["spectral", "spike"]
            objDict = load(os.path.join(tdir, container), dataclass=wanted)
            assert len(wanted) == len(objDict.keys())
            for ext in wanted:
                basename = container + "." + ext
                assert basename in objDict.keys()
                fname = objDict[basename].filename
                assert fname == os.path.join(os.path.join(tdir, container + FILE_EXT["dir"]),
                                             basename)
            del objDict

            # load multiple files from joint container via multiple dataclasses and single tag
            objDict = load(os.path.join(tdir, container), tag="analog",
                           dataclass=["analog", "analog"])
            wanted = ["2nd", "3rd"]
            for name, obj in objDict.items():
                assert isinstance(obj, AnalogData)
                inWanted = [tag in name for tag in wanted]
                assert any(inWanted)
                inWanted.pop(inWanted.index(True))
            del objDict

            # load multiple files from joint container via single dataclass and multiple tags
            objDict = load(os.path.join(tdir, container), tag=["2nd", "3rd"],
                           dataclass="analog")
            wanted = ["2nd", "3rd"]
            for name, obj in objDict.items():
                assert isinstance(obj, AnalogData)
                inWanted = [tag in name for tag in wanted]
                assert any(inWanted)
                inWanted.pop(inWanted.index(True))
            del objDict

            # load multiple files from joint container via multiple dataclasses and tags
            objDict = load(os.path.join(tdir, container), tag=["2nd", "3rd"],
                           dataclass=["analog", "analog"])
            wanted = ["2nd", "3rd"]
            for name, obj in objDict.items():
                assert isinstance(obj, AnalogData)
                inWanted = [tag in name for tag in wanted]
                assert any(inWanted)
                inWanted.pop(inWanted.index(True))
            del obj, objDict

            # invalid combinations of tag/dataclass
            with pytest.raises(SPYIOError):
                load(os.path.join(tdir, container), dataclass="analog", tag="invalid")
            with pytest.raises(SPYIOError):
                load(os.path.join(tdir, container), dataclass="spike", tag="2nd")
            with pytest.raises(SPYIOError):
                load(os.path.join(tdir, container), dataclass=["analog", "spike"],
                     tag="invalid")
            with pytest.raises(SPYIOError):
                load(os.path.join(tdir, container), dataclass=["spike", "invalid"],
                     tag="2nd")
            with pytest.raises(SPYIOError):
                load(os.path.join(tdir, container), dataclass="analog",
                     tag=["invalid", "stillinvalid"])
            with pytest.raises(SPYIOError):
                load(os.path.join(tdir, container), dataclass="spike",
                     tag=["2nd", "3rd"])
            with pytest.raises(SPYIOError):
                load(os.path.join(tdir, container), dataclass=["spike", "event"],
                     tag=["2nd", "3rd"])
            with pytest.raises(SPYIOError):
                load(os.path.join(tdir, container), dataclass=["analog", "analog"],
                     tag=["invalid", "stillinvalid"])
            with pytest.raises(SPYIOError):
                load(os.path.join(tdir, container), dataclass=["spike", "event"],
                     tag=["invalid", "stillinvalid"])
            with pytest.raises(SPYValueError):
                load(os.path.join(tdir, container), dataclass="invalid", tag="2nd")
            with pytest.raises(SPYValueError):
                load(os.path.join(tdir, container), dataclass=["invalid", "stillinvalid"],
                     tag="2nd")

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
            save(adata, filename=dname, memuse=60)
            assert (mem - memory_usage()[0]) < 70

            # Delete all open references to file objects b4 closing tmp dir
            del dmap, adata
