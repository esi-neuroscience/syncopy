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

# Local imports

from syncopy import __acme__
if __acme__:
    import dask.distributed as dd

from syncopy import AnalogData
from syncopy import connectivityanalysis as cafunc
import syncopy.tests.synth_data as synth_data
import syncopy.tests.helpers as helpers
from syncopy.shared.errors import SPYValueError
from syncopy.shared.tools import get_defaults

# Decorator to decide whether or not to run dask-related tests
skip_without_acme = pytest.mark.skipif(not __acme__, reason="acme not available")
# Decorator to decide whether or not to run memory-intensive tests
availMem = psutil.virtual_memory().total
minRAM = 5
skip_low_mem = pytest.mark.skipif(availMem < minRAM * 1024**3, reason=f"less than {minRAM}GB RAM available")


class TestGranger:

    nTrials = 100
    nChannels = 5
    nSamples = 1000
    fs = 200

    # -- Create a somewhat intricate
    # -- network of AR(2) processes

    # the adjacency matrix
    # encodes coupling strength directly
    AdjMat = np.zeros((nChannels, nChannels))
    AdjMat[0, 4] = 0.15
    AdjMat[3, 4] = 0.15
    AdjMat[3, 2] = 0.25
    AdjMat[1, 0] = 0.25

    # channel indices of coupling
    # a number other than 0 at AdjMat(i,j)
    # means coupling from i->j
    cpl_idx = np.where(AdjMat)
    nocpl_idx = np.where(AdjMat == 0)

    trls = []
    for _ in range(nTrials):
        # defaults AR(2) parameters yield 40Hz peak
        trls.append(synth_data.AR2_network(AdjMat, nSamples=nSamples))

    # create syncopy data instance
    data = AnalogData(trls, samplerate=fs)
    time_span = [-1, nSamples / fs - 1]   # -1s offset
    foi = np.arange(5, 75)   # in Hz

    def test_gr_solution(self, **kwargs):


        Gcaus = cafunc(self.data, method='granger',
                       tapsmofrq=3, foi=None, **kwargs)


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
                plot_Granger(Gcaus, i, j)
                ppl.legend()

    def test_gr_selections(self):

        # trial, channel and toi selections
        selections = helpers.mk_selection_dicts(self.nTrials,
                                                self.nChannels,
                                                *self.time_span)

        for sel_dct in selections:

            Gcaus = cafunc(self.data, method='granger', select=sel_dct)

            # check here just for finiteness and positivity
            assert np.all(np.isfinite(Gcaus.data))
            assert np.all(Gcaus.data[0, ...] >= -1e-10)

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

    @skip_without_acme
    @skip_low_mem
    def test_gr_parallel(self, testcluster=None):

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

        pad_length = 6 # seconds
        call = lambda pad: self.test_gr_solution(pad=pad)
        helpers.run_padding_test(call, pad_length)

    def test_gr_polyremoval(self):

        # add a constant to the signals
        self.data = self.data + 10

        call = lambda polyremoval: self.test_gr_solution(polyremoval=polyremoval)
        helpers.run_polyremoval_test(call)

        # remove the constant again
        self.data = self.data - 10


class TestCoherence:

    nSamples = 1500
    nChannels = 6
    nTrials = 100
    fs = 1000

    # -- two harmonics with individual phase diffusion --

    f1, f2 = 20, 40
    trls = []
    for _ in range(nTrials):
        # a lot of phase diffusion (1% per step) in the 20Hz band
        p1 = synth_data.phase_diffusion(f1, eps=.01, nChannels=nChannels, nSamples=nSamples)
        # little diffusion in the 40Hz band
        p2 = synth_data.phase_diffusion(f2, eps=0.001, nChannels=nChannels, nSamples=nSamples)
        # superposition
        signals = np.cos(p1) + np.cos(p2)
        # noise stabilizes the result(!!)
        signals += np.random.randn(nSamples, nChannels)
        trls.append(signals)

    data = AnalogData(trls, samplerate=fs)
    time_span = [-1, nSamples / fs - 1]   # -1s offset

    def test_coh_solution(self, **kwargs):

        res = cafunc(data=self.data,
                     method='coh',
                     foilim=[5, 60],
                     output='pow',
                     tapsmofrq=1.5,
                     **kwargs)

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
        assert np.all(res.data[0, null_idx, 0, 1] < 0.15)

        plot_coh(res, 0, 1, label="channel 0-1")

    def test_coh_selections(self):

        selections = helpers.mk_selection_dicts(self.nTrials,
                                                self.nChannels,
                                                *self.time_span)

        for sel_dct in selections:

            result = cafunc(self.data, method='coh', select=sel_dct)

            # check here just for finiteness and positivity
            assert np.all(np.isfinite(result.data))
            assert np.all(result.data[0, ...] >= -1e-10)

    def test_coh_foi(self):

        call = lambda foi, foilim: cafunc(self.data,
                                          method='coh',
                                          foi=foi,
                                          foilim=foilim)

        helpers.run_foi_test(call, foilim=[0, 70])

    def test_coh_cfg(self):

        call = lambda cfg: cafunc(self.data, cfg)
        run_cfg_test(call, method='coh',
                     cfg=get_defaults(cafunc))

    @skip_without_acme
    @skip_low_mem
    def test_coh_parallel(self, testcluster=None):

        ppl.ioff()
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and 'parallel' not in attr)]

        for test in all_tests:
            test_method = getattr(self, test)
            test_method()
        client.close()
        ppl.ion()

    def test_coh_padding(self):

        pad_length = 2 # seconds
        call = lambda pad: self.test_coh_solution(pad=pad)
        helpers.run_padding_test(call, pad_length)

    def test_coh_polyremoval(self):

        call = lambda polyremoval: self.test_coh_solution(polyremoval=polyremoval)
        helpers.run_polyremoval_test(call)


class TestCorrelation:

    nChannels = 5
    nTrials = 10
    fs = 1000
    nSamples = 2000 # 2s long signals

    # -- a single harmonic with phase shifts between channels

    f1 = 10   # period is 0.1s
    trls = []
    for _ in range(nTrials):

        # no phase diffusion
        p1 = synth_data.phase_diffusion(f1, eps=0, nChannels=nChannels, nSamples=nSamples)
        # same frequency but more diffusion
        p2 = synth_data.phase_diffusion(f1, eps=0.1, nChannels=1, nSamples=nSamples)
        # set 2nd channel to higher phase diffusion
        p1[:, 1] = p2[:, 0]
        # add a pi/2 phase shift for the even channels
        p1[:, 2::2] += np.pi / 2

        trls.append(np.cos(p1))

    data = AnalogData(trls, samplerate=fs)
    time_span = [-1, nSamples / fs - 1] # -1s offset

    def test_corr_solution(self, **kwargs):

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

    @skip_without_acme
    @skip_low_mem
    def test_corr_parallel(self, testcluster=None):

        ppl.ioff()
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and 'parallel' not in attr)]

        for test in all_tests:
            test_method = getattr(self, test)
            test_method()
        client.close()
        ppl.ion()

    def test_corr_polyremoval(self):

        call = lambda polyremoval: self.test_corr_solution(polyremoval=polyremoval)
        helpers.run_polyremoval_test(call)


def run_cfg_test(method_call, method, cfg, positivity=True):

    cfg.method = method
    if method != 'granger':
        cfg.foilim = [0, 70]
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
