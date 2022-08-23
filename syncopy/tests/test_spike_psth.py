# -*- coding: utf-8 -*-
#
# Test Peri-Stimulus Time Histogram
#

import numpy as np

# syncopy imports
import syncopy as spy
from syncopy.tests import synth_data as sd


# === THIS IS A STUB ===
class TestPSTH:

    # synthetic spike data
    nTrials = 10
    spd = sd.poisson_noise(nTrials,
                           nUnits=4,
                           nChannels=2,
                           nSpikes=10000,
                           samplerate=10000)

    def test_psth(self):

        cfg = spy.StructDict()
        cfg.binsize = 0.3
        cfg.latency = [-.5, 1.5]

        counts = spy.spike_psth(self.spd,
                                cfg,
                                keeptrials=True)

        assert np.allclose(np.diff(counts.time[0]), cfg.binsize)


if __name__ == '__main__':
    T1 = TestPSTH()

