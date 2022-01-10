# -*- coding: utf-8 -*-
#
# Test connectivity measures
#

# 3rd party imports
import psutil
import pytest
import inspect
import itertools
import numpy as np
import matplotlib.pyplot as ppl

# Local imports
from syncopy import __acme__
if __acme__:
    import dask.distributed as dd
from syncopy.datatype import AnalogData
from syncopy.connectivity import connectivity
import syncopy.tests.synth_data as synth_data
from syncopy.shared.errors import SPYValueError
from syncopy.shared.tools import get_defaults

# Decorator to decide whether or not to run dask-related tests
skip_without_acme = pytest.mark.skipif(not __acme__, reason="acme not available")
# Decorator to decide whether or not to run memory-intensive tests
availMem = psutil.virtual_memory().total
skip_low_mem = pytest.mark.skipif(availMem < 5 * 1024**3, reason="less than 10GB RAM available")


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

    print(AdjMat)
    # channel indices of coupling
    # a 1 at AdjMat(i,j) means coupling from i->j
    cpl_idx = np.where(AdjMat)
    nocpl_idx = np.where(AdjMat == 0)

    trls = []
    for _ in range(nTrials):
        # defaults AR(2) parameters yield 40Hz peak
        trls.append(synth_data.AR2_network(AdjMat, nSamples=nSamples))

    # create syncopy data instance
    data = AnalogData(trls, samplerate=fs)
    time_span = [-1, nSamples / fs - 1] # -1s offset
    foi = np.arange(5, 75) # in Hz

    def test_gr_solution(self):

        Gcaus = connectivity(self.data, method='granger',
                             taper='dpss', tapsmofrq=3, foi=self.foi)

        # print("peak \t Aij \t peak-frq \t ij")
        # check all channel combinations with coupling
        for i, j in zip(*self.cpl_idx):
            peak = Gcaus.data[0, :, i, j].max()
            peak_frq = Gcaus.freq[Gcaus.data[0, :, i, j].argmax()]
            cval = self.AdjMat[i, j]

            dbg_str = f"{peak:.2f}\t{self.AdjMat[i,j]:.2f}\t {peak_frq:.2f}\t"
            print(dbg_str,f'\t {i}', f' {j}')

            # test for directional coupling
            # at the right frequency range
            assert peak >= cval
            assert 35 < peak_frq < 45

            plot_Granger(Gcaus, i, j, None)
            ppl.legend()

    def test_gr_selections(self):

        selections = mk_selection_comb(self.nTrials,
                                       self.nChannels,
                                       *self.time_span)

        for sel_dct in selections:

            Gcaus = connectivity(self.data, method='granger', select=sel_dct)

            # check here just for finiteness and positivity
            assert np.all(np.isfinite(Gcaus.data))
            assert np.all(Gcaus.data[0, ...] >= -1e-10)

    def test_gr_foi(self):

        call = lambda foi, foilim: connectivity(self.data,
                                                method='granger',
                                                foi=foi,
                                                foilim=foilim)

        run_foi_test(call, foilim=[0, 70])

    def test_gr_cfg(self):

        call = lambda cfg: connectivity(self.data, cfg)
        run_cfg_test(call, method='granger')

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


class TestCoherence:

    nSamples = 1500
    nChannels = 3
    nTrials = 100
    fs = 1000

    # -- two harmonics with individual phase diffusion --

    f1, f2 = 20, 40
    trls = []
    for _ in range(nTrials):
        # a lot of phase diffusion (1% per step) in the 20Hz band
        p1 = synth_data.phase_evo(f1, eps=.01, nChannels=nChannels, nSamples=nSamples)
        # little diffusion in the 40Hz band
        p2 = synth_data.phase_evo(f2, eps=0.001, nChannels=nChannels, nSamples=nSamples)
        # superposition
        signals = np.cos(p1) + np.cos(p2)
        # noise stabilizes the result(!!)
        signals += np.random.randn(nSamples, nChannels)
        trls.append(signals)

    data = AnalogData(trls, samplerate=fs)
    time_span = [-1, nSamples / fs - 1] # -1s offset

    def test_coh_solution(self):

        res = connectivity(data=self.data,
                           method='coh',
                           foilim=[5, 60],
                           output='pow',
                           taper='dpss',
                           tapsmofrq=1)

        # coherence at the harmonic frequencies
        idx_f1 = np.argmin(res.freq < self.f1)
        peak_f1 = res.data[0, idx_f1, 0, 1]
        idx_f2 = np.argmin(res.freq < self.f2)
        peak_f2 = res.data[0, idx_f2, 0, 1]

        # check low phase diffusion has high coherence
        assert peak_f2 > 0.7
        # check that with higher phase diffusion the
        # coherence is lower
        assert peak_f1 < peak_f2

        # check that 5Hz away from the harmonics there
        # is low coherence
        null_idx = (res.freq < self.f1 - 5) | (res.freq > self.f1 + 5)
        null_idx *= (res.freq < self.f2 - 5) | (res.freq > self.f2 + 5)
        assert np.all(res.data[0, null_idx, 0, 1] < 0.1)

    def test_coh_selections(self):

        selections = mk_selection_comb(self.nTrials,
                                       self.nChannels,
                                       *self.time_span)

        for sel_dct in selections:
            print(sel_dct)
            result = connectivity(self.data, method='coh', select=sel_dct)

            # check here just for finiteness and positivity
            assert np.all(np.isfinite(result.data))
            assert np.all(result.data[0, ...] >= -1e-10)

    def test_coh_foi(self):

        call = lambda foi, foilim: connectivity(self.data,
                                                method='coh',
                                                foi=foi,
                                                foilim=foilim)

        run_foi_test(call, foilim=[0, 70])

    def test_coh_cfg(self):

        call = lambda cfg: connectivity(self.data, cfg)
        run_cfg_test(call, method='coh')

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

# -- helper functions --


def run_cfg_test(call, method):

    cfg = get_defaults(connectivity)

    cfg.method = method
    cfg.foilim = [0, 70]
    cfg.taper = 'parzen'
    cfg.output = 'abs'

    result = call(cfg)

    # check here just for finiteness and positivity
    assert np.all(np.isfinite(result.data))
    assert np.all(result.data[0, ...] >= -1e-10)


def run_foi_test(call, foilim):

    # only positive frequencies
    assert np.min(foilim) >= 0
    assert np.max(foilim) <= 500

    # fois
    foi1 = np.arange(foilim[0], foilim[1]) # 1Hz steps
    foi2 = np.arange(foilim[0], foilim[1], 0.25) # 0.5Hz steps
    foi3 = 'all'
    fois = [foi1, foi2, foi3, None]

    for foi in fois:
        # FIXME: this works for method='granger' but not method='coh' 0.0
        print(foi)
        result = call(foi=foi, foilim=None)
        # check here just for finiteness and positivity
        assert np.all(np.isfinite(result.data))
        assert np.all(result.data[0, ...] >= -1e-10)

    # 2 foilims
    foilims = [[2, 60], [7.65, 45.1234], None]
    for foil in foilims:
        result = call(foilim=foil, foi=None)
        # check here just for finiteness and positivity
        assert np.all(np.isfinite(result.data))
        assert np.all(result.data[0, ...] >= -1e-10)

    # make sure specification of both foi and foilim triggers a
    # Syncopy ValueError
    try:
        result = call(foi=foi, foilim=foil)
    except SPYValueError as err:
        assert 'foi/foilim' in str(err)

    # make sure out-of-range foi selections are detected
    try:
        result = call(foilim=[-1, 70], foi=None)
    except SPYValueError as err:
        assert 'foilim' in str(err)
        assert 'bounded by' in str(err)

    try:
        result = call(foi=np.arange(550, 700), foilim=None)
    except SPYValueError as err:
        assert 'foi' in str(err)
        assert 'bounded by' in str(err)


def mk_selection_comb(nTrials, nChannels, toi_min, toi_max):

    # at least 10 trials
    assert nTrials > 9
    # at least 2 channels
    assert nChannels > 1
    # at least 250ms
    assert (toi_max - toi_min) > 0.25

    # create 3 random trial and channel selections
    trials, channels = [], []
    for _ in range(3):

        sizeTr =  np.random.randint(10, nTrials + 1)
        trials.append(list(np.random.choice(
            nTrials, size=sizeTr
        )
        ))

        sizeCh = np.random.randint(2, nChannels + 1)
        channels.append(['channel' + str(i + 1)
                         for i in
                         np.random.choice(
                             nChannels, size=sizeCh, replace=False
                         )])

    # create toi selections, signal length is toi_max
    # with -1s as offset (from synthetic data instantiation)
    # subsampling does NOT WORK due to precision issues :/
    # toi1 = np.linspace(-.4, 2, 100)
    tois = [None, 'all']
    toi_combinations = itertools.product(trials,
                                         channels,
                                         tois)

    # 2 random toilims
    toilims = []
    while len(toilims) < 2:

        toil = np.sort(np.random.rand(2)) * (toi_max - toi_min) + toi_min
        # at least 250ms
        if np.diff(toil) < 0.25:
            continue
        else:
            toilims.append(toil)

    # combinatorics of all selection options
    # order matters to assign the selection dict keys!
    toilim_combinations = itertools.product(trials,
                                            channels,
                                            toilims)

    selections = []
    # digest generators to create all selection dictionaries
    for comb in toi_combinations:

        sel_dct = {}
        sel_dct['trials'] = comb[0]
        sel_dct['channels'] = comb[1]
        sel_dct['toi'] = comb[2]
        selections.append(sel_dct)

    for comb in toilim_combinations:

        sel_dct = {}
        sel_dct['trials'] = comb[0]
        sel_dct['channels'] = comb[1]
        sel_dct['toilim'] = comb[2]
        selections.append(sel_dct)

    return selections


def plot_Granger(G, i, j, color='cornflowerblue'):

    ax = ppl.gca()
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'Granger causality(f)')
    ax.plot(G.freq, G.data[0, :, i, j], label=f'Granger {i}-{j}',
            alpha=0.7, lw=1.3, c=color)
    ax.set_ylim((-.1, 1.3))


def plot_coh(res, i, j, label=''):

    ax = ppl.gca()
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('coherence $|CSD|^2$')
    ax.plot(res.freq, res.data[0, :, i, j], label=label)


if __name__ == '__main__':
    T1 = TestGranger()
    T2 = TestCoherence()
