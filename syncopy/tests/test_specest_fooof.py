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
    nChannels = 2
    nChan2 = int(nChannels / 2)
    nTrials = 1
    seed = 151120
    fadeIn = None
    fadeOut = None
    tfData, modulators, even, odd, fader = _make_tf_signal(nChannels, nTrials, seed,
                                                           fadeIn=fadeIn, fadeOut=fadeOut, short=True)
    cfg = get_defaults(freqanalysis)
    cfg.method = "mtmfft"
    cfg.taper = "hann"
    cfg.select = {"trials": 0, "channel": 1}
    cfg.output = "fooof"

    def test_spfooof_output_fooof(self, fulltests):
        self.cfg['output'] = "fooof"
        spec_dt = freqanalysis(self.cfg, self.tfData)
        assert spec_dt.data.ndim == 4
        # TODO: add meaningful tests here

    def test_spfooof_output_fooof_aperiodic(self, fulltests):                
        self.cfg['output'] = "fooof_aperiodic"
        spec_dt = freqanalysis(self.cfg, self.tfData)
        assert spec_dt.data.ndim == 4
        # TODO: add meaningful tests here

    def test_spfooof_output_fooof_peaks(self, fulltests):                
        self.cfg['output'] = "fooof_peaks"
        spec_dt = freqanalysis(self.cfg, self.tfData)
        assert spec_dt.data.ndim == 4
        # TODO: add meaningful tests here
