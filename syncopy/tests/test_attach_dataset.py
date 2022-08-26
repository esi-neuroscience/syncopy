# -*- coding: utf-8 -*-
#
# Test ability to attach extra datasets to Syncopy Data objects.
#

import numpy as np
import h5py

# syncopy imports
import syncopy as spy
from syncopy.tests import synth_data as sd
from syncopy.tests.test_spike_psth import get_spike_data, get_spike_cfg

class TestAttachDataset:

    cfg = get_spike_cfg()

    def test_attach_to_spikedata(self):
        """
        Test that we can run attach an extra sequential dataset to Syncopy SpikeData Object.
        """
        spkd = get_spike_data()

        extra_data = np.zeros((3, 3), dtype=np.float64)
        spkd._register_seq_dataset("dset_mean", extra_data)

        assert hasattr(spkd, "_dset_mean")
        assert isinstance(spkd._dset_mean, h5py.Dataset)

    def test_destruction(self):
        """
        Test destructor: there should be no exceptions/errors on destruction.
        """
        def some_local_func():
            spkd = get_spike_data()
            extra_data = np.zeros((3, 3), dtype=np.float64)
            spkd._register_seq_dataset("dset_mean", extra_data)
            # Let spkd get out of scope to call destructor.

        some_local_func()
        assert not 'spkd' in locals()

    def test_comparison(self):
        """
        Test comparison operator.
        """
        spkd1 = get_spike_data()
        spkd2 = spkd1.copy()

        assert spkd1 == spkd2

        extra_data = np.zeros((3, 3), dtype=np.float64)
        spkd2._register_seq_dataset("dset_mean", extra_data)

        assert spkd1 != spkd2

    def test_run_psth_with_attached_dset(self):
        """
        Test that we can run a cF on a Syncopy Data Object without any
        side effects, i.e., the cF should just run and leave the extra dataset alone.
        """

        spkd = get_spike_data()

        extra_data = np.zeros((3, 3), dtype=np.float64)
        spkd._register_seq_dataset("dset_mean", extra_data)

        counts = spy.spike_psth(spkd,
                                self.cfg,
                                keeptrials=True)

        # Make sure we did not interfere with the PSTH computation.
        assert np.allclose(np.diff(counts.time[0]), self.cfg.binsize)


if __name__ == '__main__':
    T1 = TestAttachDataset()