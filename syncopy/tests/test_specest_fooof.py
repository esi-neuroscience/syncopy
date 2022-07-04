import pytest
import numpy as np


from syncopy.tests.test_specest import _make_tf_signal

# Local imports
from syncopy import freqanalysis
from syncopy.shared.tools import get_defaults


class TestFOOOF():

    # FOOOF is a post-processing of an FFT, so we first generate a signal and
    # run an FFT on it. Then we run FOOOF. The first part of these tests is
    # therefore very similar to the code in TestMTMConvol above.
    #
    # Construct high-frequency signal modulated by slow oscillating cosine and
    # add time-decaying noise
    nChannels = 6
    nChan2 = int(nChannels / 2)
    nTrials = 3
    seed = 151120
    fadeIn = None
    fadeOut = None
    tfData, modulators, even, odd, fader = _make_tf_signal(nChannels, nTrials, seed,
                                                           fadeIn=fadeIn, fadeOut=fadeOut)

    def test_spfooof_output(self, fulltests):
        # Set up basic TF analysis parameters to not slow down things too much
        cfg = get_defaults(freqanalysis)
        cfg.method = "mtmfft"
        cfg.taper = "hann"
        cfg.select = {"trials" : 0, "channel" : 1}
        cfg.output = "fooof"
        tfSpec = freqanalysis(cfg, self.tfData)
        assert 1 == 1


