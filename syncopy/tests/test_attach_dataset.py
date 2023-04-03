# -*- coding: utf-8 -*-
#
# Test ability to attach extra datasets to Syncopy Data objects.
#

import numpy as np
import h5py
import pytest
import tempfile
import os

# syncopy imports
import syncopy as spy
from syncopy.tests.test_spike_psth import get_spike_data, get_spike_cfg
from syncopy.tests.test_metadata import _get_fooof_signal
from syncopy.shared.errors import SPYValueError


class TestAttachDataset:

    cfg = get_spike_cfg()

    def test_attach_to_spikedata(self):
        """
        Test that we can run attach an extra sequential dataset to Syncopy SpikeData Object.
        """
        spkd = get_spike_data()
        assert isinstance(spkd, spy.SpikeData)

        extra_data = np.zeros((3, 3), dtype=np.float64)
        spkd._register_dataset("dset_mean", extra_data)

        assert hasattr(spkd, "_dset_mean")
        assert isinstance(spkd._dset_mean, h5py.Dataset)
        assert isinstance(spkd._dset_mean.file, h5py.File)
        assert np.array_equal(spkd._dset_mean[()], extra_data)

    def test_attach_and_update(self):
        """
        Test that we can run attach an extra sequential dataset to Syncopy SpikeData Object
        and later update it with an ndarray of identical dimensions.
        """
        spkd = get_spike_data()

        extra_data = np.zeros((3, 3), dtype=np.float64)
        spkd._register_dataset("dset_mean", extra_data)

        assert hasattr(spkd, "_dset_mean")
        assert isinstance(spkd._dset_mean, h5py.Dataset)
        assert isinstance(spkd._dset_mean.file, h5py.File)
        assert np.array_equal(spkd._dset_mean[()], extra_data)

        extra_data2 = np.zeros((3, 3), dtype=np.float64) + 2

        spkd._register_dataset("dset_mean", extra_data2)
        assert hasattr(spkd, "_dset_mean")
        assert isinstance(spkd._dset_mean, h5py.Dataset)
        assert isinstance(spkd._dset_mean.file, h5py.File)
        assert np.array_equal(spkd._dset_mean[()], extra_data2)




    def test_destruction(self):
        """
        Test destructor: there should be no exceptions/errors on destruction.
        """
        def some_local_func():
            spkd = get_spike_data()
            extra_data = np.zeros((3, 3), dtype=np.float64)
            spkd._register_dataset("dset_mean", extra_data)
            # Let spkd get out of scope to call destructor.

        some_local_func()
        assert not 'spkd' in locals()

    def test_comparison_with_and_without_extra_dset(self):
        """
        Test comparison operator: if one instance has an extra dataset, they should not be equal.
        """
        spkd1 = get_spike_data()
        spkd2 = spkd1.copy()

        assert spkd1 == spkd2

        extra_data = np.zeros((3, 3), dtype=np.float64)
        spkd2._register_dataset("dset_mean", extra_data)

        assert spkd1 != spkd2

    def test_copy(self):
        """
        Test copy: copying should copy the extra attribute and dataset.
        """
        spkd1 = get_spike_data()
        extra_data = np.zeros((3, 3), dtype=np.float64)
        spkd1._register_dataset("dset_mean", extra_data)
        assert isinstance(spkd1._dset_mean.file, h5py.File)
        assert np.array_equal(spkd1._dset_mean[()], extra_data)

        spkd2 = spkd1.copy()

        assert hasattr(spkd2, "_dset_mean")
        assert isinstance(spkd2._dset_mean, h5py.Dataset)
        assert isinstance(spkd2._dset_mean.file, h5py.File)
        assert np.array_equal(spkd2._dset_mean[()], extra_data)

        assert spkd1._data.file != spkd2._data.file

    def test_comparison_of_values(self):
        """
        Test more details of equality.
        """

        spkd1 = get_spike_data()
        spkd2 = spkd1.copy()
        spkd3 = spkd1.copy()

        # Copies should be equal.
        assert spkd1 == spkd2
        assert spkd1 == spkd3

        extra_data1 = np.zeros((3, 3), dtype=np.float64)
        spkd1._register_dataset("dset_mean", extra_data1)

        extra_data2 = np.zeros((3, 4), dtype=np.float64)
        spkd2._register_dataset("dset_mean", extra_data2)

        # Copies, with different extra seq data attached to them after copying, should NOT be equal.
        assert spkd1 != spkd2

        # Note that copies, with *identical* extra seq data attached to them after copying,
        # are also not equal. This is due to the implementation of h5py.Dataset equality in h5py, which
        # is based on the `id` of the dataset. See https://github.com/h5py/h5py/blob/master/h5py/_hl/base.py#L348
        # We show this here, and one should keep it in mind:
        spkd3._register_dataset("dset_mean", extra_data1)
        assert spkd3 != spkd1  # Even though they are copies, with identical `np.ndarrays` attached as `h5py.Datasets`!

    def test_detach(self):
        """
        Test that we can attach and detach an extra sequential dataset to Syncopy SpikeData Object.
        """
        spkd = get_spike_data()

        extra_data = np.zeros((3, 3), dtype=np.float64)
        spkd._register_dataset("dset_mean", extra_data)

        assert hasattr(spkd, "_dset_mean")
        assert isinstance(spkd._dset_mean, h5py.Dataset)
        assert isinstance(spkd._dset_mean.file, h5py.File)
        assert np.array_equal(spkd._dset_mean[()], extra_data)

        spkd._unregister_dataset("dset_mean", del_from_file=False)
        assert not hasattr(spkd, "_dset_mean")
        assert "dset_mean" in h5py.File(spkd.filename, "r").keys()
        assert "data" in h5py.File(spkd.filename, "r").keys()

        spkd._unregister_dataset("dset_mean", del_from_file=True)
        assert not hasattr(spkd, "_dset_mean")
        assert not "dset_mean" in h5py.File(spkd.filename, "r").keys()
        assert "data" in h5py.File(spkd.filename, "r").keys()

    def test_with_analog_data(self):
        """
        Test that we can run attach, update and detach an extra sequential
        dataset to Syncopy AnalogData Object.
        """

        def some_local_func(data1, data2):

            adt = _get_fooof_signal()
            assert isinstance(adt, spy.AnalogData)

            # Copying
            adt2 = adt.copy()

            adt._register_dataset("dset_mean", data1)

            # Equality testing
            assert adt != adt2

            assert hasattr(adt, "_dset_mean")
            assert isinstance(adt._dset_mean, h5py.Dataset)
            assert isinstance(adt._dset_mean.file, h5py.File)
            assert np.array_equal(adt._dset_mean[()], data1)

            # Update
            adt._update_dataset("dset_mean", data2)

            # Unregister
            adt._unregister_dataset("dset_mean", del_from_file=True)
            assert not hasattr(adt, "_dset_mean")
            assert not "dset_mean" in h5py.File(adt.filename, "r").keys()
            assert "data" in h5py.File(adt.filename, "r").keys()
            # Let it get out of scope to call destructor.
            del adt
            del adt2

        extra_data1 = np.zeros((3, 3), dtype=np.float64)
        extra_data2 = np.zeros((3, 3), dtype=np.float64) + 2
        some_local_func(extra_data1, extra_data2)
        assert 'adt' not in locals()

        # repeat with hdf5 datasets
        tfile1 = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        tfile1.close()
        tfile2 = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        tfile2.close()
        with h5py.File(tfile1.name, 'w') as file1:
            extra_ds1 = file1.create_dataset("d1", extra_data1.shape)
            extra_ds1[()] = extra_data1

            with h5py.File(tfile2.name, 'w') as file2:
                extra_ds2 = file2.create_dataset("d2", extra_data2.shape)
                extra_ds2[()] = extra_data2

                some_local_func(extra_ds1, extra_ds2)
                assert 'adt' not in locals()

        tfile1.close()
        os.unlink(tfile1.name)
        tfile2.close()
        os.unlink(tfile2.name)

    def test_attach_None_to_analog_data(self):
        """
        Test that we can run attach, update and detach an extra sequential
        dataset with None data to Syncopy AnalogData Object.
        """
        def some_local_func():
            adt = _get_fooof_signal()
            assert isinstance(adt, spy.AnalogData)

            # Copying
            adt2 = adt.copy()

            adt._register_dataset("dset_mean", None)

            # Equality testing
            assert adt != adt2

            assert hasattr(adt, "_dset_mean")
            assert adt._dset_mean is None

            # Update
            adt._register_dataset("dset_mean", None)
            assert hasattr(adt, "_dset_mean")
            assert adt._dset_mean is None

            # Unregister
            adt._unregister_dataset("dset_mean", del_from_file=True)
            assert not hasattr(adt, "_dset_mean")
            assert not "dset_mean" in h5py.File(adt.filename, "r").keys()
            assert "data" in h5py.File(adt.filename, "r").keys()
            # Let it get out of scope to call destructor.

        some_local_func()
        assert not 'adt' in locals()

    def test_run_psth_with_attached_dset(self):
        """
        Test that we can run a cF on a Syncopy Data Object without any
        side effects, i.e., the cF should just run and leave the extra dataset alone.

        We do NOT expect the cF to interact with the extra dataset in any way! This also
        means that the extra dataset will NOT show up in the result of the cF (the output SyncopyData object),
        as that is a new object and we would explicitely need to move the extra dataset over to it
        in 'process_metadata'.
        """
        spkd = get_spike_data()

        extra_data = np.zeros((3, 3), dtype=np.float64)
        spkd._register_dataset("dset_mean", extra_data)

        counts = spy.spike_psth(spkd,
                                self.cfg,
                                keeptrials=True)

        # Make sure we did not interfere with the PSTH computation.
        assert np.allclose(np.diff(counts.time[0]), self.cfg.binsize)

        # Make sure the extra data set is there.
        assert hasattr(spkd, "_dset_mean")

        # Make clear that we do NOT expect the extra dataset in the output.
        assert not hasattr(counts, "_dset_mean")

    def test_errors_reattach_wrong_shape_or_type(self):
        """Test that errors are raised when we try to update with data with different dims."""

        spkd = get_spike_data()
        extra_data = np.zeros((3, 3), dtype=np.float64)
        spkd._register_dataset("dset_mean", extra_data)

        extra_data_diff_type = np.zeros((3, 3), dtype=np.int32)
        with pytest.raises(SPYValueError, match="dataset of type"):
            spkd._register_dataset("dset_mean", extra_data_diff_type)

        extra_data_diff_shape = np.zeros((3, 3, 5), dtype=np.float64)
        with pytest.raises(SPYValueError, match="dataset with shape"):
            spkd._register_dataset("dset_mean", extra_data_diff_shape)

        # Show that we can delete the old one, and attach a new one with different shape.
        spkd._unregister_dataset("dset_mean")
        spkd._register_dataset("dset_mean", extra_data_diff_type) # Fine this time, it's new.

    def test_save_load_unregister(self):
        """Test that saving and loading with attached seq datasets works.
           Also tests that the attached datasets gets deleted from the
           backing HDF5 file when calling `_unregister_dataset()`.
        """

        spkd = get_spike_data()
        extra_data = np.zeros((3, 3), dtype=np.float64)
        spkd._register_dataset("dset_mean", extra_data)


        tfile0 = tempfile.NamedTemporaryFile(suffix=".spike", delete=True)
        tfile0.close()
        # Test save and load.
        tmp_spy_filename = tfile0.name
        spy.save(spkd, filename=tmp_spy_filename)
        spkd2 = spy.load(filename=tmp_spy_filename)
        assert isinstance(spkd2._dset_mean, h5py.Dataset)
        assert np.array_equal(spkd._dset_mean[()], spkd2._dset_mean[()])

        # Test delete/unregister.
        spkd2._unregister_dataset("dset_mean")
        assert "dset_mean" not in h5py.File(tmp_spy_filename, mode="r").keys()
        tfile0.close()

        spkd = get_spike_data()

        # repeat with hdf5 datasets
        tfile1 = tempfile.NamedTemporaryFile(suffix=".spike", delete=False)
        tfile1.close()
        tfile2 = tempfile.NamedTemporaryFile(suffix=".spike", delete=True)
        tfile2.close()

        file1 = h5py.File(tfile1.name, 'w')
        extra_dset = file1.create_dataset("d1", extra_data.shape)
        extra_dset[()] = extra_data

        spkd._register_dataset("dset_mean", extra_dset)

        # Test save and load.
        tmp_spy_filename = tfile2.name
        spy.save(spkd, filename=tmp_spy_filename)
        spkd2 = spy.load(filename=tmp_spy_filename)
        assert isinstance(spkd2._dset_mean, h5py.Dataset)
        assert np.array_equal(spkd._dset_mean[()], spkd2._dset_mean[()])

        # Test delete/unregister.
        spkd2._unregister_dataset("dset_mean")
        tfile2.close()
        assert "dset_mean" not in h5py.File(tmp_spy_filename, mode="r").keys()


if __name__ == '__main__':
    T1 = TestAttachDataset()
