# -*- coding: utf-8 -*-
#
# Test Peri-Stimulus Time Histogram
#

import numpy as np

# syncopy imports
import syncopy as spy
from syncopy.tests import synth_data as sd

def get_spike_data(nTrials = 10):
    return sd.poisson_noise(nTrials,
                           nUnits=4,
                           nChannels=2,
                           nSpikes=10000,
                           samplerate=10000)
def get_spike_cfg():
    cfg = spy.StructDict()
    cfg.binsize = 0.3
    cfg.latency = [-.5, 1.5]
    return cfg


# === THIS IS A STUB ===
class TestPSTH:

    # synthetic spike data
    spd = get_spike_data()

    def test_psth(self):

        cfg = get_spike_cfg()

        counts = spy.spike_psth(self.spd,
                                cfg,
                                keeptrials=True)

        assert np.allclose(np.diff(counts.time[0]), cfg.binsize)


if __name__ == '__main__':
    T1 = TestPSTH()

