# -*- coding: utf-8 -*-
#
# Test connectivity measures
#

# 3rd party imports
import psutil
import pytest
import inspect
import numpy as np
import matplotlib.pyplot as ppl
import dask.distributed as dd

# Local imports

import syncopy as spy
from syncopy import AnalogData, SpectralData
from syncopy.connectivity.connectivity_analysis import connectivity_outputs
from syncopy import connectivityanalysis as cafunc
import syncopy.tests.synth_data as synth_data
import syncopy.tests.helpers as helpers
from syncopy.shared.errors import SPYValueError
from syncopy.shared.tools import get_defaults

# Decorator to decide whether or not to run memory-intensive tests
availMem = psutil.virtual_memory().total
minRAM = 5
skip_low_mem = pytest.mark.skipif(availMem < minRAM * 1024**3, reason=f"less than {minRAM}GB RAM available")


class TestSpectralInput:
    """
    When inputting SpectralData directly into connectivityanalysis, it has to fulfill
    certain conditions: be complex (output='fourier'), multi-trial (no premature trial averaging)
    and in a multi-taper setting, the tapers can't be averaged before.
    """

    # mockup data
    ad = AnalogData([np.ones((5, 10)) for _ in range(2)], samplerate=200)

    def test_spectral_output(self):
        for wrong_output in ['pow', 'abs', 'imag', 'real']:
            spec = spy.freqanalysis(self.ad, output=wrong_output)

            with pytest.raises(SPYValueError) as err:
                cafunc(spec, method='granger')
            assert "expected complex valued" in str(err.value)

            with pytest.raises(SPYValueError) as err:
                cafunc(spec, method='coh')
            assert "expected complex valued" in str(err.value)

    def test_spectral_multitaper(self):

        # default with needed output='fourier' does not work already in freqanalysis
        # -> taper averaging with keeptapers=False not meaningful with fourier output
        with pytest.raises(SPYValueError) as err:
            spec = spy.freqanalysis(self.ad, tapsmofrq=0.1, output='fourier')
        assert "expected 'pow'|False" in str(err.value)

        # single trial /  trial averaging makes no sense
        spec = spy.freqanalysis(self.ad, tapsmofrq=0.1, keeptrials=False,
                                output='fourier', keeptapers=True)
        with pytest.raises(SPYValueError) as err:
            cafunc(spec, method='coh')
        assert "expected multi-trial input data" in str(err.value)

    def test_spectral_corr(self):

        # method='corr' does not work with SpectralData
        spec = spy.freqanalysis(self.ad, tapsmofrq=0.1,
                                output='fourier', keeptapers=True)
        with pytest.raises(SPYValueError) as err:
            cafunc(spec, method='corr')
        assert "expected AnalogData" in str(err.value)

    def test_tf_input(self):
        """ No time-resolved Granger implemented yet """
        spec = spy.freqanalysis(self.ad, method='mtmconvol',
                                t_ftimwin=0.01, output='fourier')

        with pytest.raises(NotImplementedError) as err:
            cafunc(spec, method='granger')
        assert "Granger causality from tf-spectra" in str(err.value)


class TestGranger:

    nTrials = 200
    nChannels = 4
    nSamples = 500
    fs = 200

    # -- Create a somewhat intricate
    # -- network of AR(2) processes

    # the adjacency matrix
    # encodes coupling strength directly
    AdjMat = np.zeros((nChannels, nChannels))
    AdjMat[3, 1] = 0.15
    AdjMat[3, 2] = 0.25
    AdjMat[1, 0] = 0.25

    # channel indices of coupling
    # a number other than 0 at AdjMat(i,j)
    # means coupling from i->j
    cpl_idx = np.where(AdjMat)
    nocpl_idx = np.where(AdjMat == 0)

    data = synth_data.AR2_network(nTrials,
                                  AdjMat=AdjMat,
                                  nSamples=nSamples,
                                  samplerate=fs,
                                  seed=42)

    time_span = [-1, nSamples / fs - 1]   # -1s offset

    cfg = spy.StructDict()
    cfg.tapsmofrq = 1
    cfg.foi = None
    spec = spy.freqanalysis(data, cfg, output='fourier', keeptapers=True, demean_taper=True)

    def test_spec_input_frontend(self):
        assert isinstance(self.spec, SpectralData)
        cfg = self.cfg.copy()
        cfg.pop("tapsmofrq", None)
        res = spy.connectivityanalysis(self.spec, method='granger', cfg=cfg)
        assert isinstance(res, spy.CrossSpectralData)

    def test_gr_solution(self, **kwargs):

        # re-run spectral analysis
        if len(kwargs) != 0:
            spec = spy.freqanalysis(self.data, self.cfg, output='fourier',
                                    keeptapers=True, demean_taper=True, **kwargs)
        else:
            spec = self.spec

        # sanity check
        assert isinstance(self.data, AnalogData)
        assert isinstance(spec, SpectralData)

        # from SpectralData
        Gcaus_spec = cafunc(spec, method='granger', **kwargs)

        # from AnalogData directly, needs cfg for spectral analyis
        Gcaus_ad = cafunc(self.data, method='granger',
                          cfg=self.cfg, **kwargs)

        # same results on all channels and freqs within 2%
        assert np.allclose(Gcaus_ad.trials[0], Gcaus_spec.trials[0], atol=2e-2)

        for Gcaus in [Gcaus_spec, Gcaus_ad]:
            # check all channel combinations with coupling
            for i, j in zip(*self.cpl_idx):
                peak = Gcaus.data[0, :, i, j].max()
                peak_frq = Gcaus.freq[Gcaus.data[0, :, i, j].argmax()]
                cval = self.AdjMat[i, j]

                dbg_str = f"{peak:.2f}\t{self.AdjMat[i,j]:.2f}\t {peak_frq:.2f}\t"
                print(dbg_str, f'\t {i}', f' {j}')

                # test for directional coupling
                # at the right frequency range
                assert peak >= cval
                assert 35 < peak_frq < 45

                # only plot with defaults
                if len(kwargs) == 0:
                    Gcaus.singlepanelplot(channel_i=i, channel_j=j)
                    # plot_Granger(Gcaus, i, j)
                    # ppl.legend()

            # check .info for default test
            if len(kwargs) == 0:
                assert Gcaus.info['converged']
                assert Gcaus.info['max rel. err'] < 1e-5
                assert Gcaus.info['reg. factor'] == 0
                assert Gcaus.info['initial cond. num'] > 10

            # Test that 'metadata_keys' in the Granger ComputationalRoutine is up-to-date. All listed
            #  keys should exist...
            for k in spy.connectivity.AV_compRoutines.GrangerCausality.metadata_keys:
                assert k in Gcaus.info
            # ... and no unmentioned extra keys should be in there.
            assert len(Gcaus.info) == len(spy.connectivity.AV_compRoutines.GrangerCausality.metadata_keys)

    def test_gr_selections(self):

        # trial, channel and toi selections
        selections = helpers.mk_selection_dicts(self.nTrials,
                                                self.nChannels,
                                                *self.time_span)

        for sel_dct in selections:
            print(sel_dct)
            Gcaus_ad = cafunc(self.data, self.cfg,
                              method='granger', select=sel_dct)

            # selections act on spectral analysis, remove latency
            # sel_dct.pop('latency')
            spec = spy.freqanalysis(self.data, self.cfg, output='fourier',
                                    keeptapers=True, select=sel_dct, demean_taper=True)

            Gcaus_spec = cafunc(spec, method='granger')

            # check here just for finiteness and positivity
            assert np.all(np.isfinite(Gcaus_ad.data))
            assert np.all(Gcaus_ad.data[0, ...] >= -1e-10)

            # same results
            assert np.allclose(Gcaus_ad.trials[0], Gcaus_spec.trials[0], atol=1e-2)

        # test one final selection into a result
        # obtained via orignal SpectralData input
        selections[0].pop('latency')
        result_ad = cafunc(self.data, self.cfg, method='granger', select=selections[0])
        result_spec = cafunc(self.spec, method='granger', select=selections[0])
        assert np.allclose(result_ad.trials[0], result_spec.trials[0], atol=2e-2)

    def test_gr_foi(self):

        try:
            cafunc(self.data,
                   method='granger',
                   foi=np.arange(0, 70)
                   )
        except SPYValueError as err:
            assert 'no foi specification' in str(err)

        try:
            cafunc(self.data,
                   method='granger',
                   foilim=[0, 70]
                   )
        except SPYValueError as err:
            assert 'no foi specification' in str(err)

    def test_gr_cfg(self):

        call = lambda cfg: cafunc(self.data, cfg)
        run_cfg_test(call, method='granger',
                     cfg=get_defaults(cafunc))

    @skip_low_mem
    def test_gr_parallel(self, testcluster):

        ppl.ioff()
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and 'parallel' not in attr)]

        for test in all_tests:
            test_method = getattr(self, test)
            test_method()
        client.close()
        ppl.ion()

    def test_gr_padding(self):

        pad_length = 6   # seconds
        call = lambda pad: self.test_gr_solution(pad=pad)
        helpers.run_padding_test(call, pad_length)

    def test_gr_polyremoval(self):

        call = lambda polyremoval: self.test_gr_solution(polyremoval=polyremoval)
        helpers.run_polyremoval_test(call)


class TestCoherence:

    nSamples = 1400
    nChannels = 4
    nTrials = 100
    fs = 1000

    # -- two harmonics with individual phase diffusion --

    f1, f2 = 20, 40
    # a lot of phase diffusion (1% per step) in the 20Hz band
    s1 = synth_data.phase_diffusion(nTrials, freq=f1,
                                    eps=.03,
                                    nChannels=nChannels,
                                    nSamples=nSamples,
                                    seed=helpers.test_seed)

    # little diffusion in the 40Hz band
    s2 = synth_data.phase_diffusion(nTrials, freq=f2,
                                    eps=.001,
                                    nChannels=nChannels,
                                    nSamples=nSamples,
                                    seed=helpers.test_seed)

    wn = synth_data.white_noise(nTrials, nChannels=nChannels, nSamples=nSamples,
                                seed=helpers.test_seed)

    # superposition
    data = s1 + s2 + wn
    data.samplerate = fs
    time_span = [-1, nSamples / fs - 1]   # -1s offset

    # spectral analysis
    cfg = spy.StructDict()
    cfg.tapsmofrq = 1.5
    cfg.foilim = [5, 60]

    spec = spy.freqanalysis(data, cfg, output='fourier', keeptapers=True)

    def test_timedep_coh(self):
        """
        Time dependent coherence of phase diffusing signals.
        Starting from a common phase, they'll decorrelate over time.
        """

        # 20Hz band has strong diffusion so coherence
        # will go down noticably over the observation time
        test_data = self.data
        # check number of samples
        nSamples = self.nSamples
        # assert test_data.time[0].size == nSamples

        # get time-frequency spec for the non-stationary signal
        spec_tf = spy.freqanalysis(test_data, method='mtmconvol',
                                   t_ftimwin=0.3, foilim=[5, 100],
                                   output='fourier')

        # check that we have still the same time axis
        assert np.all(spec_tf.time[0] == test_data.time[0])

        # abs averaged spectra for plotting
        spec_tf_abs = spy.SpectralData(data=np.abs(spec_tf.data),
                                       samplerate=self.fs,
                                       trialdefinition=spec_tf.trialdefinition)
        spec_tf_abs.freq = spec_tf.freq
        spec_tf_abs.multipanelplot(trials=0)

        # rough check that the power in the 20Hz band is lower than in the 40Hz
        # band due to more phase diffusion
        spec_tf_avabs = spy.mean(spec_tf_abs, dim='trials')
        profile_20 = spec_tf_avabs.show(frequency=self.f1)[:, 1]
        profile_40 = spec_tf_avabs.show(frequency=self.f2)[:, 1]

        assert profile_40.mean() > profile_20.mean()
        assert profile_40.max() > profile_20.max()

        # compute time dependent coherence
        coh = cafunc(data=spec_tf, method='coh')

        # check that we have still the same time axis
        assert np.all(coh.time[0] == test_data.time[0])

        # not exactly beautiful but it makes the point
        coh.singlepanelplot(channel_i=0, channel_j=1, frequency=[7, 60])

        # plot the coherence over time just along three different frequency bands
        ppl.figure()
        cprofile20 = coh.show(frequency=self.f1, channel_i=0, channel_j=1)
        ppl.plot(cprofile20, label='20Hz')

        # coherence goes down more slowly
        cprofile40 = coh.show(frequency=self.f2, channel_i=0, channel_j=1)
        ppl.plot(cprofile40, label='40Hz')

        # here is nothing
        cprofile10 = coh.show(frequency=10, channel_i=0, channel_j=1)
        ppl.plot(cprofile10, label='10Hz')
        ppl.xlabel('samples')
        ppl.ylabel('coherence')
        ppl.legend()

        # check that the 20 Hz band has high coherence only in the beginning
        assert cprofile20.max() > 0.9
        # later coherence goes down
        assert cprofile20[int(0.9 * nSamples):].max() < 0.4
        # side band never has high coherence except for the very beginning
        assert cprofile10.max() < 0.5

    def test_coh_solution(self, **kwargs):

        # re-run spectral analysis
        if len(kwargs) != 0:
            spec = spy.freqanalysis(self.data, self.cfg, output='fourier',
                                    keeptapers=True, **kwargs)
        else:
            spec = self.spec
        # sanity check
        assert isinstance(self.data, AnalogData)
        assert isinstance(spec, SpectralData)

        res_spec = cafunc(data=spec,
                          method='coh',
                          **kwargs)

        # needs same cfg for spectral analysis
        res_ad = cafunc(data=self.data,
                        method='coh',
                        cfg=self.cfg,
                        **kwargs)

        # same results on all channels and freqs
        assert np.allclose(res_spec.trials[0], res_ad.trials[0])

        for res in [res_spec, res_ad]:
            # coherence at the harmonic frequencies
            idx_f1 = np.argmin(res.freq < self.f1)
            peak_f1 = res.data[0, idx_f1, 0, 1]
            idx_f2 = np.argmin(res.freq < self.f2)
            peak_f2 = res.data[0, idx_f2, 0, 1]

            # check low phase diffusion has high coherence
            assert peak_f2 > 0.5
            # check that with higher phase diffusion the
            # coherence is lower
            assert peak_f1 < peak_f2

            # check that 5Hz away from the harmonics there
            # is low coherence
            null_idx = (res.freq < self.f1 - 5) | (res.freq > self.f1 + 5)
            null_idx *= (res.freq < self.f2 - 5) | (res.freq > self.f2 + 5)
            assert np.all(res.data[0, null_idx, 0, 1] < 0.2)

            if len(kwargs) == 0:
                res.singlepanelplot(channel_i=0, channel_j=1)

    def test_coh_selections(self):

        selections = helpers.mk_selection_dicts(self.nTrials,
                                                self.nChannels,
                                                *self.time_span)

        for sel_dct in selections:
            result = cafunc(self.data, self.cfg, method='coh', select=sel_dct)

            # re-run spectral analysis with selection
            spec = spy.freqanalysis(self.data, self.cfg, output='fourier',
                                    keeptapers=True, select=sel_dct)

            result_spec = cafunc(spec, method='coh')

            # check here just for finiteness and positivity
            assert np.all(np.isfinite(result.data))
            assert np.all(result.data[0, ...] >= -1e-10)

            # same results
            assert np.allclose(result.trials[0], result_spec.trials[0], atol=1e-3)

        # test one final selection into a result
        # obtained via orignal SpectralData input
        selections[0].pop('latency')
        result_ad = cafunc(self.data, self.cfg, method='coh', select=selections[0])
        result_spec = cafunc(self.spec, method='coh', select=selections[0])
        assert np.allclose(result_ad.trials[0], result_spec.trials[0], atol=1e-3)

    def test_coh_foi(self):

        # 2 frequencies
        foilim = [[2, 60], [7.65, 45.1234], None]
        for foil in foilim:
            result = cafunc(self.data, method='coh', foilim=foil)
            # check here just for finiteness and positivity
            assert np.all(np.isfinite(result.data))
            assert np.all(result.data[0, ...] >= -1e-10)

        # make sure out-of-range foilim  are detected
        with pytest.raises(SPYValueError, match='foilim'):
            result = cafunc(self.data, method='coh', foilim=[-1, 70])

        # make sure invalid foilim are detected
        with pytest.raises(SPYValueError, match='foilim'):
            result = cafunc(self.data, method='coh', foilim=[None, None])

        with pytest.raises(SPYValueError, match='foilim'):
            result = cafunc(self.data, method='coh', foilim='abc')

    def test_coh_cfg(self):

        call = lambda cfg: cafunc(self.data, cfg)
        run_cfg_test(call, method='coh',
                     cfg=get_defaults(cafunc))

    @skip_low_mem
    def test_coh_parallel(self, testcluster):
        check_parallel(self, testcluster)

    def test_coh_padding(self):

        pad_length = 2   # seconds
        call = lambda pad: self.test_coh_solution(pad=pad)
        helpers.run_padding_test(call, pad_length)

    def test_coh_polyremoval(self):

        call = lambda polyremoval: self.test_coh_solution(polyremoval=polyremoval)
        helpers.run_polyremoval_test(call)

    def test_coh_outputs(self):

        for output in connectivity_outputs:
            coh = cafunc(self.data,
                         method='coh',
                         output=output)

            if output in ['complex', 'fourier']:
                # we have imaginary parts
                assert not np.all(np.imag(coh.trials[0]) == 0)
            elif output == 'angle':
                # all values in [-pi, pi]
                assert np.all((coh.trials[0] < np.pi) | (coh.trials[0] > -np.pi))
            else:
                # strictly real outputs
                assert np.all(np.imag(coh.trials[0]) == 0)


class TestCSD:
    nSamples = 1400
    nChannels = 4
    nTrials = 100
    fs = 1000
    Method = 'csd'

    # -- two harmonics with individual phase diffusion --

    f1, f2 = 20, 40
    # a lot of phase diffusion (1% per step) in the 20Hz band
    s1 = synth_data.phase_diffusion(nTrials, freq=f1,
                                    eps=.01,
                                    nChannels=nChannels,
                                    nSamples=nSamples,
                                    seed=42)

    # little diffusion in the 40Hz band
    s2 = synth_data.phase_diffusion(nTrials, freq=f2,
                                    eps=.001,
                                    nChannels=nChannels,
                                    nSamples=nSamples,
                                    seed=42)

    wn = synth_data.white_noise(nTrials, nChannels=nChannels, nSamples=nSamples)

    # superposition
    data = s1 + s2 + wn
    data.samplerate = fs
    time_span = [-1, nSamples / fs - 1]   # -1s offset

    # spectral analysis
    cfg = spy.StructDict()
    cfg.tapsmofrq = 1.5
    cfg.foilim = [5, 60]

    spec = spy.freqanalysis(data, cfg, output='fourier', keeptapers=True)

    def test_data_output_type(self):
        cross_spec = spy.connectivityanalysis(self.spec, method='csd')
        assert np.all(self.spec.freq == cross_spec.freq)
        assert cross_spec.data.dtype.name == 'complex64'
        assert cross_spec.data.shape != self.spec.data.shape

    @skip_low_mem
    def test_csd_parallel(self, testcluster):
        check_parallel(self, testcluster)

    def test_csd_input(self):
        assert isinstance(self.spec, SpectralData)

    def test_csd_cfg_replay(self):
        cross_spec = spy.connectivityanalysis(self.spec, method=self.Method)
        assert len(cross_spec.cfg) == 2
        assert np.all([True for cfg in zip(self.spec.cfg['freqanalysis'], cross_spec.cfg['freqanalysis']) if cfg[0] == cfg[1]])
        assert cross_spec.cfg['connectivityanalysis'].method == self.Method

        first_cfg = cross_spec.cfg['connectivityanalysis']
        first_res = spy.connectivityanalysis(self.spec, cfg=first_cfg)
        replay_res = spy.connectivityanalysis(self.spec, cfg=first_res.cfg)

        assert np.allclose(first_res.data[:], replay_res.data[:])
        assert first_res.cfg == replay_res.cfg


class TestCorrelation:

    nChannels = 5
    nTrials = 50
    fs = 1000
    nSamples = 2001   # 2s long signals

    # -- a single harmonic with phase shifts between channels

    f1 = 10   # period is 0.1s
    trls = []
    for _ in range(nTrials):

        # no phase diffusion
        p1 = synth_data.phase_diffusion(freq=f1,
                                        eps=0,
                                        nChannels=nChannels,
                                        nSamples=nSamples,
                                        seed=42,
                                        return_phase=True)
        # same frequency but more diffusion
        p2 = synth_data.phase_diffusion(freq=f1,
                                        eps=0.1,
                                        nChannels=1,
                                        nSamples=nSamples,
                                        seed=42,
                                        return_phase=True)

        # set 2nd channel to higher phase diffusion
        p1[:, 1] = p2[:, 0]
        # add a pi/2 phase shift for the even channels
        p1[:, 2::2] += np.pi / 2

        trls.append(np.cos(p1))

    data = AnalogData(trls, samplerate=fs)
    time_span = [-1, nSamples / fs - 1]  # -1s offset

    def test_corr_solution(self, **kwargs):

        # `keeptrials=False` is the default here!
        corr = cafunc(data=self.data, method='corr', **kwargs)

        # test 0-lag autocorr is 1 for all channels
        assert np.all(corr.data[0, 0].diagonal() > .99)

        # test that at exactly the period-lag
        # correlations remain high w/o phase diffusion
        period_idx = int(1 / self.f1 * self.fs)
        # 100 samples is one period
        assert np.allclose(100, period_idx)
        auto_00 = corr.data[:, 0, 0, 0]
        assert np.all(auto_00[::period_idx] > .99)

        # test for auto-corr minima at half the period
        assert auto_00[period_idx // 2] < -.99
        assert auto_00[period_idx // 2 + period_idx] < -.99

        # test signal with phase diffusion (2nd channel) has
        # decaying correlations (diffusion may lead to later
        # increases of auto-correlation again, hence we check
        # only the first 5 periods)
        auto_11 = corr.data[:, 0, 1, 1]
        assert np.all(np.diff(auto_11[::period_idx])[:5] < 0)

        # test that a pi/2 phase shift moves the 1st
        # crosscorr maximum to 1/4 of the period
        cross_02 = corr.data[:, 0, 0, 2]
        lag_idx = int(1 / self.f1 * self.fs * 0.25)
        # 25 samples is 1/4th period
        assert np.allclose(25, lag_idx)
        assert cross_02[lag_idx] > 0.99
        # same for a period multiple
        assert cross_02[lag_idx + period_idx] > .99
        # plus half the period a minimum occurs
        assert cross_02[lag_idx + period_idx // 2] < -.99

        # test for (anti-)symmetry
        cross_20 = corr.data[:, 0, 2, 0]
        assert cross_20[-lag_idx] > 0.99
        assert cross_20[-lag_idx - period_idx] > 0.99

        # only plot for simple solution test
        if len(kwargs) == 0:

            # test that keeptrials=False yields (almost) same results
            # as post-hoc trial averaging
            corr_st = cafunc(data=self.data, method='corr', keeptrials=True)
            corr_st_trl_avg = spy.mean(corr_st, dim='trials')
            # keeptrials=False normalizes with global signal variances
            # hence there can be actually small differences
            assert np.allclose(corr_st_trl_avg.data[()], corr.data[()], atol=1e-2)

            plot_corr(corr, 0, 0, label='corr 0-0')
            plot_corr(corr, 1, 1, label='corr 1-1')
            plot_corr(corr, 0, 2, label='corr 0-2')
            ppl.xlim((-.01, 0.5))
            ppl.ylim((-1.1, 1.3))
            ppl.legend(ncol=3)

    def test_corr_padding(self):

        self.test_corr_solution(pad='maxperlen')
        # no padding is allowed for
        # this method
        try:
            self.test_corr_solution(pad=1000)
        except SPYValueError as err:
            assert 'pad' in str(err)
            assert 'no padding needed/allowed' in str(err)

        try:
            self.test_corr_solution(pad='nextpow2')
        except SPYValueError as err:
            assert 'pad' in str(err)
            assert 'no padding needed/allowed' in str(err)

        try:
            self.test_corr_solution(pad='IamNoPad')
        except SPYValueError as err:
            assert 'Invalid value of `pad`' in str(err)
            assert 'no padding needed/allowed' in str(err)

    def test_corr_selections(self):

        selections = helpers.mk_selection_dicts(self.nTrials,
                                                self.nChannels,
                                                *self.time_span)

        for sel_dct in selections:

            result = cafunc(self.data, method='corr', select=sel_dct)

            # check here just for finiteness and positivity
            assert np.all(np.isfinite(result.data))

    def test_corr_cfg(self):

        call = lambda cfg: cafunc(self.data, cfg)
        run_cfg_test(call, method='corr',
                     positivity=False,
                     cfg=get_defaults(cafunc))

    @skip_low_mem
    def test_corr_parallel(self, testcluster):
        check_parallel(self, testcluster)

    def test_corr_polyremoval(self):

        call = lambda polyremoval: self.test_corr_solution(polyremoval=polyremoval)
        helpers.run_polyremoval_test(call)


class TestPPC:

    nSamples = 1000
    nChannels = 3
    nTrials = 20
    fs = 1000

    # -- one harmonic with individual phase diffusion --

    f1 = 20
    # phase diffusion (1% per step) in the 20Hz band
    s1 = synth_data.phase_diffusion(nTrials, freq=f1,
                                    eps=.01,
                                    nChannels=nChannels,
                                    nSamples=nSamples,
                                    seed=helpers.test_seed)
    wn = synth_data.white_noise(nTrials, nChannels=nChannels, nSamples=nSamples,
                                seed=helpers.test_seed)

    # superposition
    data = s1 + wn
    data.samplerate = fs
    time_span = [-1, nSamples / fs - 1]   # -1s offset

    # spectral analysis
    cfg = spy.StructDict()
    cfg.tapsmofrq = 1.5
    cfg.foilim = [5, 60]

    spec = spy.freqanalysis(data, cfg, output='fourier', keeptapers=True)

    def test_timedep_ppc(self):
        """
        Time dependent PPC of phase diffusing signals.
        Starting from a common phase, they'll decorrelate over time.
        """

        # 20Hz band has strong diffusion so coherence
        # will go down noticably over the observation time
        test_data = self.data
        # check number of samples
        assert test_data.time[0].size == self.nSamples

        # get time-frequency spec for the non-stationary signal
        spec_tf = spy.freqanalysis(test_data, method='mtmconvol',
                                   t_ftimwin=0.3, foilim=[5, 100],
                                   output='fourier')

        # compute time dependent coherence
        ppc = cafunc(data=spec_tf, method='ppc')

        # check that we have still the same time axis
        assert np.all(ppc.time[0] == test_data.time[0])

        # not exactly beautiful but it makes the point
        ppc.singlepanelplot(channel_i=0, channel_j=1, frequency=[7, 60])

        # for visual comparison to the coherence which has more bias
        coh = cafunc(data=spec_tf, method='coh')

        # plot the coherence over time just along two different frequency bands
        ppl.figure()
        ppc_profile20 = ppc.show(frequency=self.f1, channel_i=0, channel_j=1)
        ppl.plot(ppc_profile20, label='20Hz')
        ppl.plot(coh.show(frequency=20, channel_i=0, channel_j=1), ls='--',
                 label='20Hz coherence', c='k', alpha=0.4)

        # here is nothing
        ppc_profile50 = ppc.show(frequency=50, channel_i=0, channel_j=1)
        ppl.plot(ppc_profile50, label='50Hz')

        ppl.plot(coh.show(frequency=50, channel_i=0, channel_j=1),
                 label='50Hz coherence', c='k', alpha=0.4)

        ppl.title(f'PPC(t), nTrials={self.nTrials}')
        ppl.xlabel('samples')
        ppl.ylabel('PPC')
        ppl.legend()

        # check that the 20 Hz band has high PPC only in the beginning
        assert ppc_profile20.max() > 0.9
        assert ppc_profile20.argmax() < self.nSamples / 10
        # side band never has high PPC
        # note that we can go lower as compared to the coherence
        # as PPC has less bias
        assert ppc_profile50.max() < 0.2

        # check that for the sideband noise (0 true coherence) we get lower values as
        # for the classic coherence
        assert np.all(coh.show(frequency=50, channel_i=0, channel_j=1) > ppc_profile50)

    def test_ppc_solution(self, **kwargs):

        # re-run spectral analysis
        if len(kwargs) != 0:
            spec = spy.freqanalysis(self.data, self.cfg, output='fourier',
                                    keeptapers=True, **kwargs)
        else:
            spec = self.spec
        # sanity check
        assert isinstance(self.data, AnalogData)
        assert isinstance(spec, SpectralData)

        res_spec = cafunc(data=spec,
                          method='ppc',
                          **kwargs)

        # needs same cfg for spectral analysis
        res_ad = cafunc(data=self.data,
                        method='ppc',
                        cfg=self.cfg,
                        **kwargs)

        # same results on all channels and freqs
        # irrespective of AnalogData or SpectralData input
        assert np.allclose(res_spec.trials[0], res_ad.trials[0])

        for res in [res_spec, res_ad]:
            # coherence at the harmonic frequencies
            idx_f1 = np.argmin(res.freq < self.f1)
            peak_f1 = res.data[0, idx_f1, 0, 1]

            assert peak_f1 > 0.25
            # check that highest coherence is at the
            # harmonic frequency
            # assert res.show(channel_i=0, channel_j=1).argmax() == idx_f1

            # check that 5Hz away from the harmonics there
            # is low PPC
            null_idx = (res.freq < self.f1 - 5) | (res.freq > self.f1 + 5)
            assert np.all(res.data[0, null_idx, 0, 1] < 0.2)

            if len(kwargs) == 0:
                res.singlepanelplot(channel_i=0, channel_j=1)

    def test_ppc_selections(self):

        sel_dct = helpers.mk_selection_dicts(self.nTrials,
                                             self.nChannels,
                                             *self.time_span)[0]

        result = cafunc(self.data, self.cfg, method='coh', select=sel_dct)

        # re-run spectral analysis with selection
        spec = spy.freqanalysis(self.data, self.cfg, output='fourier',
                                keeptapers=True, select=sel_dct)

        result_spec = cafunc(spec, method='coh')

        # check here just for finiteness and positivity
        assert np.all(np.isfinite(result.data))
        assert np.all(result.data[0, ...] >= -1e-10)

        # same results
        assert np.allclose(result.trials[0], result_spec.trials[0], atol=1e-3)

        # test one final selection into a result
        # obtained via orignal SpectralData input
        sel_dct.pop('latency')
        result_ad = cafunc(self.data, self.cfg, method='ppc', select=sel_dct)
        result_spec = cafunc(self.spec, method='ppc', select=sel_dct)
        assert np.allclose(result_ad.trials[0], result_spec.trials[0], atol=1e-3)

    def test_ppc_foi(self):

        # make sure out-of-range foilim  are detected
        with pytest.raises(SPYValueError, match='foilim'):
            _ = cafunc(self.data, method='ppc', foilim=[-1, 70])

        # make sure invalid foilim are detected
        with pytest.raises(SPYValueError, match='foilim'):
            _ = cafunc(self.data, method='ppc', foilim=[None, None])

        with pytest.raises(SPYValueError, match='foilim'):
            _ = cafunc(self.data, method='ppc', foilim='abc')

    @skip_low_mem
    def test_ppc_parallel(self, testcluster):
        check_parallel(self, testcluster)

    def test_ppc_padding(self):

        pad_length = 2   # seconds
        call = lambda pad: self.test_ppc_solution(pad=pad)
        helpers.run_padding_test(call, pad_length)

    def test_ppc_polyremoval(self):

        call = lambda polyremoval: self.test_ppc_solution(polyremoval=polyremoval)
        helpers.run_polyremoval_test(call)


def check_parallel(TestClass, testcluster):
    ppl.ioff()
    client = dd.Client(testcluster)
    all_tests = [attr for attr in TestClass.__dir__()
                 if (inspect.ismethod(getattr(TestClass, attr)) and 'parallel' not in attr)]
    for test in all_tests:
        test_method = getattr(TestClass, test)
        test_method()
    client.close()
    ppl.ion()


def run_cfg_test(method_call, method, cfg, positivity=True):

    cfg.method = method
    if method != 'granger':
        cfg.frequency = [0, 70]
    # test general tapers with
    # additional parameters
    cfg.taper = 'kaiser'
    cfg.taper_opt = {'beta': 2}

    cfg.output = 'abs'

    result = method_call(cfg)

    # check here just for finiteness and positivity
    assert np.all(np.isfinite(result.data))
    if positivity:
        assert np.all(result.data[0, ...] >= -1e-10)


def plot_Granger(G, i, j):

    ax = ppl.gca()
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'Granger causality(f)')
    ax.plot(G.freq, G.data[0, :, i, j], label=f'Granger {i}-{j}',
            alpha=0.7, lw=1.3)
    ax.set_ylim((-.1, 1.3))


def plot_coh(res, i, j, label=''):

    ax = ppl.gca()
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('coherence $|CSD|^2$')
    ax.plot(res.freq, res.data[0, :, i, j], label=label)
    ax.legend()


def plot_corr(res, i, j, label=''):

    ax = ppl.gca()
    ax.set_xlabel('lag (s)')
    ax.set_ylabel('Correlation')
    ax.plot(res.time[0], res.data[:, 0, i, j], label=label)
    ax.legend()


if __name__ == '__main__':
    T1 = TestGranger()
    T2 = TestCoherence()
    T3 = TestCorrelation()
    T4 = TestSpectralInput()
    T5 = TestCSD()
    T6 = TestPPC()
