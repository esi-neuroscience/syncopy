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
from glob import glob
import matplotlib.pyplot as ppl

# Local imports
import syncopy as spy
from syncopy.datatype import AnalogData
from syncopy.io import save, load, load_ft_raw, load_tdt, load_nwb
from syncopy.shared.filetypes import FILE_EXT
from syncopy.shared.errors import (
    SPYValueError,
    SPYIOError,
    SPYError,
    SPYTypeError
)
import syncopy.datatype as swd
from syncopy.tests.misc import generate_artificial_data



# Decorator to detect if test data dir is available
on_esi = os.path.isdir('/cs/slurm/syncopy')
skip_no_esi = pytest.mark.skipif(not on_esi, reason="ESI fs not available")
skip_no_nwb = pytest.mark.skipif(not spy.__nwb__, reason="pynwb not installed")

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

    # Generate a 4D array simulating a CorssSpectralData array
    data["CrossSpectralData"] = np.arange(1, nc * nc * ns * nf + 1).reshape(ns, nf, nc, nc)
    trl["CrossSpectralData"] = trl["AnalogData"]

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
    classes = ["AnalogData", "SpectralData", "CrossSpectralData", "SpikeData", "EventData"]

    # Test correct handling of object log
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

            # ensure loading is logged correctly
            dummy2 = load(filename=fname + ".analog")
            assert len(dummy2._log) > len(dummy._log)
            assert dummy2.filename in dummy2._log
            assert dummy2.filename + FILE_EXT["info"] in dummy._log

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


@skip_no_esi
class TestFTImporter:

    """At the moment only ft_datatype_raw is supported"""

    mat_file_dir = '/cs/slurm/syncopy/MAT-Files'

    def test_read_hdf(self):
        """Test MAT-File v73 reader, uses h5py"""

        mat_name = 'matdata-v73.mat'
        fname = os.path.join(self.mat_file_dir, mat_name)

        dct = load_ft_raw(fname)
        assert 'Data_K' in dct
        AData = dct['Data_K']

        assert isinstance(AData, AnalogData)
        assert len(AData.trials) == 393
        assert len(AData.channel) == 218

        # list only structure names
        slist = load_ft_raw(fname, list_only=True)
        assert 'Data_K' in slist
        assert 'Data_KB' in slist

        # additional fields of Matlab structures
        # get attached to .info dict
        # hdf reader does NOT support nested fields
        dct = load_ft_raw(fname, include_fields=('chV1',))
        AData2 = dct['Data_K']
        assert 'chV1' in AData2.info
        assert len(AData2.info['chV1']) == 30
        assert isinstance(AData2.info['chV1'][0], str)

        # test loading a subset of structures
        dct = load_ft_raw(fname, select_structures=('Data_KB',))
        assert 'Data_KB' in dct
        assert 'Data_K' not in dct

        # test str sequence parsing
        try:
            dct = load_ft_raw(fname, select_structures=(3, 'sth'))
        except SPYTypeError as err:
            assert 'expected str found int' in str(err)

        try:
            dct = load_ft_raw(fname, include_fields=(3, 'sth'))
        except SPYTypeError as err:
            assert 'expected str found int' in str(err)

    def test_read_dict(self):
        """Test MAT-File v7 reader, based on scipy.io.loadmat"""

        mat_name = 'matdataK-v7.mat'
        fname = os.path.join(self.mat_file_dir, mat_name)

        dct = load_ft_raw(fname)
        assert 'Data_K' in dct
        AData = dct['Data_K']

        assert isinstance(AData, AnalogData)
        assert len(AData.trials) == 393
        assert len(AData.channel) == 218

        slist = load_ft_raw(fname, list_only=True)
        assert 'Data_K' in slist

        # additional fields of Matlab structures
        # get attached to .info dict
        # here nested structures are also now forbidden
        dct = load_ft_raw(fname, include_fields=('ch',))
        AData2 = dct['Data_K']
        # sadly here it is actually nested
        assert len(AData2.info) == 0


@skip_no_esi
class TestTDTImporter:

    tdt_dir = '/cs/slurm/syncopy/Tdt_reader/session-25'
    start_code, end_code = 23000, 30020

    def test_load_tdt(self):

        AData = load_tdt(self.tdt_dir)

        assert isinstance(AData, AnalogData)
        # check meta info parsing
        assert len(AData.info.keys()) == 13
        # that is apparently fixed
        assert AData.dimord == ['time', 'channel']
        assert len(AData.channel) == 9

        # it's only one big trial here
        assert AData.trials[0].shape == (3170560, 9)

        # test median subtr
        AData2 = load_tdt(self.tdt_dir, subtract_median=True)
        assert np.allclose(np.median(AData2.data), 0)

        # check that it wasn't 0 before
        assert not np.allclose(np.median(AData.data), 0)

        # test automatic trialdefinition
        AData = load_tdt(self.tdt_dir, self.start_code, self.end_code)
        assert len(AData.trials) == 659

    def test_exceptions(self):

        with pytest.raises(SPYIOError, match='Cannot read'):
            load_tdt('non/existing/path')

        with pytest.raises(SPYValueError, match='Invalid value of `start_code`'):
            load_tdt(self.tdt_dir, start_code=None, end_code=self.end_code)

        with pytest.raises(SPYValueError, match='Invalid value of `end_code`'):
            load_tdt(self.tdt_dir, start_code=self.start_code, end_code=None)

        with pytest.raises(SPYValueError, match='Invalid value of `start_code`'):
            load_tdt(self.tdt_dir, start_code=999999, end_code=self.end_code)

        with pytest.raises(SPYValueError, match='Invalid value of `end_code`'):
            load_tdt(self.tdt_dir, start_code=self.start_code, end_code=999999)


@skip_no_esi
@skip_no_nwb
class TestNWBImporter:

    nwb_filename = '/cs/slurm/syncopy/NWBdata/test.nwb'

    def test_load_nwb(self):

        spy_filename = self.nwb_filename.split('/')[-1][:-4] + '.spy'
        out = load_nwb(self.nwb_filename, memuse=2000)
        edata, adata1, adata2 = list(out.values())

        assert isinstance(adata2, spy.AnalogData)
        assert isinstance(edata, spy.EventData)
        assert np.any(~np.isnan(adata2.data))
        assert np.any(adata2.data != 0)

        snippet = adata2.selectdata(latency=[30, 32])

        snippet.singlepanelplot(latency=[30, 30.3], channel=3)
        ppl.gcf().suptitle('raw data')

        # Bandpass filter
        lfp = spy.preprocessing(snippet, filter_class='but', freq=[10, 100],
                                filter_type='bp', order=8)

        # Downsample
        lfp = spy.resampledata(lfp, resamplefs=2000, method='downsample')
        lfp.info = adata2.info
        lfp.singlepanelplot(channel=3)
        ppl.gcf().suptitle('bp-filtered 10-100Hz and resampled')

        spec = spy.freqanalysis(lfp, foilim=[5, 150])
        spec.singlepanelplot(channel=[1, 3])
        ppl.gcf().suptitle('bp-filtered 10-100Hz and resampled')

        # test save and load
        with tempfile.TemporaryDirectory() as tdir:
            lfp.save(os.path.join(tdir, spy_filename))
            lfp2 = spy.load(os.path.join(tdir, spy_filename))

            assert np.allclose(lfp.data, lfp2.data)


if __name__ == '__main__':
    T1 = TestFTImporter()
    T2 = TestTDTImporter()
    T3 = TestNWBImporter()
