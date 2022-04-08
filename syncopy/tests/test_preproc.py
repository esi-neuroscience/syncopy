# -*- coding: utf-8 -*-
#
# Test preprocessing
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

from syncopy import preprocessing as ppfunc
from syncopy import AnalogData, freqanalysis
import syncopy.preproc as preproc  # submodule
import syncopy.tests.helpers as helpers

from syncopy.shared.errors import SPYValueError
from syncopy.shared.tools import get_defaults, best_match

# Decorator to decide whether or not to run dask-related tests
skip_without_acme = pytest.mark.skipif(not __acme__, reason="acme not available")
# Decorator to decide whether or not to run memory-intensive tests
availMem = psutil.virtual_memory().total
minRAM = 5
skip_low_mem = pytest.mark.skipif(availMem < minRAM * 1024**3, reason=f"less than {minRAM}GB RAM available")

# availableFilterTypes = ('lp', 'hp', 'bp', 'bs')
# availableDirections = ('twopass', 'onepass', 'onepass-minphase')
# availableWindows = ("hamming", "hann", "blackman")


class TestButterworth:

    nSamples = 1000
    nChannels = 4
    nTrials = 100
    fs = 200
    fNy = fs / 2

    # -- use flat white noise as test data --

    trls = []
    for _ in range(nTrials):
        trl = np.random.randn(nSamples, nChannels)
        trls.append(trl)

    data = AnalogData(trls, samplerate=fs)
    # for toi tests, -1s offset
    time_span = [-.5, 3.1]
    flow, fhigh = 0.3 * fNy, 0.4 * fNy
    freq_kw = {'lp': fhigh, 'hp': flow,
               'bp': [flow, fhigh], 'bs': [flow, fhigh]}

    def test_but_filter(self, **kwargs):

        """
        We test for remaining power after filtering
        for all available filter types.
        Minimum order is 4 to safely pass..
        """
        # check if we run the default test
        def_test = not len(kwargs)

        # write default parameters dict
        if def_test:
            kwargs = {'direction': 'twopass',
                      'order': 4}

        # the unfiltered data
        spec = freqanalysis(self.data, tapsmofrq=3, keeptrials=False)
        # total power in arbitrary units (for now)
        pow_tot = spec.show(channel=0).sum()
        nFreq = spec.freq.size

        if def_test:
            fig, ax = mk_spec_ax()

        for ftype in preproc.availableFilterTypes:
            filtered = ppfunc(self.data,
                              filter_class='but',
                              filter_type=ftype,
                              freq=self.freq_kw[ftype],
                              **kwargs)

            # check in frequency space
            spec_f = freqanalysis(filtered, tapsmofrq=3, keeptrials=False)

            # get relevant frequency ranges
            # for integrated powers
            if ftype == 'lp':
                foilim = [0, self.freq_kw[ftype]]
            elif ftype == 'hp':
                # toilim selections can screw up the
                # frequency axis of freqanalysis/np.fft.rfftfreq :/
                foilim = [self.freq_kw[ftype], spec_f.freq[-1]]
            else:
                foilim = self.freq_kw[ftype]

            # remaining power after filtering
            pow_fil = spec_f.show(channel=0, foilim=foilim).sum()
            _, idx = best_match(spec_f.freq, foilim, span=True)
            # ratio of pass-band to total freqency band
            ratio = len(idx) / nFreq

            # at least 80% of the ideal filter power
            # should be still around
            if ftype in ('lp', 'hp'):
                assert 0.8 * ratio < pow_fil / pow_tot
            # here we have two roll-offs, one at each side
            elif ftype == 'bp':
                assert 0.7 * ratio < pow_fil / pow_tot
            # as well as here
            elif ftype == 'bs':
                assert 0.7 * ratio < (pow_tot - pow_fil) / pow_tot
            if def_test:
                plot_spec(ax, spec_f, label=ftype)

        # plotting
        if def_test:
            plot_spec(ax, spec, c='0.3', label='unfiltered')
            annotate_foilims(ax, *self.freq_kw['bp'])
            ax.set_title(f"Twopass Butterworth, order = {kwargs['order']}")

    def test_but_kwargs(self):

        """
        Test order and direction parameter
        """

        for direction in preproc.availableDirections:
            kwargs = {'direction': direction,
                      'order': 4}
            # only for firws
            if 'minphase' in direction:
                try:
                    self.test_but_filter(**kwargs)
                except SPYValueError as err:
                    assert "expected 'onepass'" in str(err)

        for order in [-2, 10, 5.6]:
            kwargs = {'direction': 'twopass',
                      'order': order}

            if order < 1 and isinstance(order, int):
                try:
                    self.test_but_filter(**kwargs)
                except SPYValueError as err:
                    assert "value to be greater" in str(err)

            else:
                try:
                    self.test_but_filter(**kwargs)
                except SPYValueError as err:
                    assert "expected int_like" in str(err)

    def test_but_selections(self):

        sel_dicts = helpers.mk_selection_dicts(nTrials=20,
                                               nChannels=2,
                                               toi_min=self.time_span[0],
                                               toi_max=self.time_span[1],
                                               min_len=2)
        for sd in sel_dicts:
            self.test_but_filter(select=sd)

    def test_but_polyremoval(self):

        helpers.run_polyremoval_test(self.test_but_filter)

    def test_but_cfg(self):

        cfg = get_defaults(ppfunc)

        cfg.filter_class = 'but'
        cfg.order = 6
        cfg.direction = 'twopass'
        cfg.freq = 30
        cfg.filter_type = 'hp'

        result = ppfunc(self.data, cfg)

        # check here just for finiteness
        assert np.all(np.isfinite(result.data))

    @skip_without_acme
    def test_but_parallel(self, testcluster=None):

        ppl.ioff()
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and 'parallel' not in attr)]

        for test_name in all_tests:
            test_method = getattr(self, test_name)
            if 'but_filter' in test_name:
                # test parallelisation along channels
                test_method(chan_per_worker=2)
            else:
                test_method()
        client.close()
        ppl.ion()


def mk_spec_ax():

    fig, ax = ppl.subplots()
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('power (dB)')
    return fig, ax


def plot_spec(ax, spec, **pkwargs):

    ax.plot(spec.freq, spec.show(channel=1), alpha=0.8, **pkwargs)
    ax.legend()


def annotate_foilims(ax, flow, fhigh):

    ylim = ax.get_ylim()
    ax.plot([flow, flow], [0, 1], 'k--')
    ax.plot([fhigh, fhigh], [0, 1], 'k--')
    ax.set_ylim(ylim)


if __name__ == '__main__':
    T1 = TestButterworth()
