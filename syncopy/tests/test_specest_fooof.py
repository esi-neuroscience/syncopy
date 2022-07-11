# -*- coding: utf-8 -*-
#
# Test FOOOF integration from user/frontend perspective.

import pytest
import numpy as np

# Local imports
from syncopy import freqanalysis
from syncopy.shared.tools import get_defaults
from syncopy.datatype import SpectralData
from syncopy.shared.errors import SPYValueError
from syncopy.tests.test_specest import _make_tf_signal


class TestFooofSpy():

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

    def test_fooof_output_fooof_fails_with_freq_zero(self, fulltests):
        self.cfg['output'] = "fooof"
        self.cfg['foilim'] = [0., 250.]    # Include the zero in tfData.
        with pytest.raises(SPYValueError) as err:
            spec_dt = freqanalysis(self.cfg, self.tfData)  # tfData contains zero.
            assert "a frequency range that does not include zero" in str(err)

    def test_fooof_output_fooof_works_with_freq_zero_in_data_after_setting_foilim(self, fulltests):
        self.cfg['output'] = "fooof"
        self.cfg['foilim'] = [0.5, 250.]    # Exclude the zero in tfData.
        spec_dt = freqanalysis(self.cfg, self.tfData)

        # check frequency axis
        assert spec_dt.freq.size == 500
        assert spec_dt.freq[0] == 0.5
        assert spec_dt.freq[499] == 250.

        # check the log
        assert "fooof_method = fooof" in spec_dt._log
        assert "fooof_aperiodic" not in spec_dt._log
        assert "fooof_peaks" not in spec_dt._log
        assert "fooof_opt" in spec_dt._log

        # check the data
        assert spec_dt.data.ndim == 4
        assert spec_dt.data.shape == (1, 1, 500, 1)
        assert not np.isnan(spec_dt.data).any()

        # Plot it.
        #spec_dt.singlepanelplot()

    def test_spfooof_output_fooof_aperiodic(self, fulltests):
        self.cfg['output'] = "fooof_aperiodic"
        assert self.cfg['foilim'] == [0.5, 250.]
        spec_dt = freqanalysis(self.cfg, self.tfData)

        # log
        assert "fooof" in spec_dt._log  # from the method
        assert "fooof_method = fooof_aperiodic" in spec_dt._log
        assert "fooof_peaks" not in spec_dt._log

        # check the data
        assert spec_dt.data.ndim == 4
        assert spec_dt.data.shape == (1, 1, 500, 1)
        assert not np.isnan(spec_dt.data).any()

    def test_spfooof_output_fooof_peaks(self, fulltests):
        self.cfg['output'] = "fooof_peaks"
        spec_dt = freqanalysis(self.cfg, self.tfData)
        assert spec_dt.data.ndim == 4
        assert "fooof" in spec_dt._log
        assert "fooof_method = fooof_peaks" in spec_dt._log
        assert "fooof_aperiodic" not in spec_dt._log


    def test_spfooof_frontend_settings_are_merged_with_defaults_used_in_backend(self, fulltests):
        self.cfg['output'] = "fooof_peaks"
        self.cfg.pop('fooof_opt', None)  # Remove from cfg to avoid passing twice. We could also modify it (and then leave out the fooof_opt kw below).
        fooof_opt = {'max_n_peaks': 8}
        spec_dt = freqanalysis(self.cfg, self.tfData, fooof_opt=fooof_opt)

        assert spec_dt.data.ndim == 4

        # TODO: test whether the settings returned as 2nd return value include
        #  our custom value for fooof_opt['max_n_peaks']. Not possible yet on
        #  this level as we have no way to get the 'details' return value.
        #  This is verified in backend tests though.

    def test_foofspy_rejects_preallocated_output(self, fulltests):
        with pytest.raises(SPYValueError) as err:
            out = SpectralData(dimord=SpectralData._defaultDimord)
            _ = freqanalysis(self.cfg, self.tfData, out=out)
            assert "pre-allocated output object not supported with" in str(err)
