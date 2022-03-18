# -*- coding: utf-8 -*-
#
# Test preprocessing 
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

from syncopy import preprocessing as pp
from syncopy import AnalogData, freqanalysis
import syncopy.preproc as preproc  # submodule
import syncopy.tests.synth_data as synth_data
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.shared.tools import get_defaults

# Decorator to decide whether or not to run dask-related tests
skip_without_acme = pytest.mark.skipif(not __acme__, reason="acme not available")
# Decorator to decide whether or not to run memory-intensive tests
availMem = psutil.virtual_memory().total
minRAM = 5
skip_low_mem = pytest.mark.skipif(availMem < minRAM * 1024**3, reason=f"less than {minRAM}GB RAM available")

#availableFilterTypes = ('lp', 'hp', 'bp', 'bs')
#availableDirections = ('twopass', 'onepass', 'onepass-minphase')
#availableWindows = ("hamming", "hann", "blackman")


class TestButterworth:

    nSamples = 1000
    nChannels = 2
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
    time_span = [-.5, .1]
    flow, fhigh = 0.3 * fNy, 0.4 * fNy
    freq_kw = {'lp': fhigh, 'hp': flow,
               'bp': [flow, fhigh], 'bs': [flow, fhigh]}

    def test_filter(self):

        fig, ax = mk_spec_ax()
        for ftype in preproc.availableFilterTypes:
            filtered = pp(self.data,
                          filter_class='but',
                          filter_type=ftype,
                          freq=self.freq_kw[ftype],
                          direction='twopass')
            # check in frequency space
            spec = freqanalysis(filtered, tapsmofrq=3, keeptrials=False)
            if ftype == 'lp':
                foilim = [0, self.freq_kw[ftype]]
            elif ftype == 'hp':
                foilim = [self.freq_kw[ftype], self.fNy]
            else:
                foilim = self.freq_kw[ftype]

            plot_spec(ax, spec, label=ftype, lw=1.5)

        # finally the unfiltered data
        spec = freqanalysis(self.data, tapsmofrq=3, keeptrials=False)
        print('unfi', spec.show(channel=1).sum())
        # plotting
        plot_spec(ax, spec, c='0.3', label='unfiltered')
        annotate_foilims(ax, *self.freq_kw['bp'])
        ax.set_title("Twopass Butterworth, order = 4")

        print(spec.show(channel=1, foilim=foilim).sum(), ftype)

    def test_filter_comb(self):

        call = lambda ftype, direction, order: pp(self.data,
                                                  filter_class='but',
                                                  filter_type=ftype,
                                                  freq=self.freq_kw[ftype],
                                                  direction=direction,
                                                  order=order)
        fig, ax = mk_spec_ax()
        for ftype in preproc.availableFilterTypes:
            for direction in preproc.availableDirections:
                for order in [2, 20]:
                    # only for firws
                    if 'minphase' in direction:
                        try:
                            call(ftype, direction, order)
                        except SPYValueError as err:
                            assert "expected 'onepass'" in str(err)
                            continue

                    filtered = call(ftype, direction, order)
                    # check in frequency space
                    spec = freqanalysis(filtered, tapsmofrq=3, keeptrials=False)
                    if ftype == 'lp':
                        foilim = [0, self.freq_kw[ftype]]
                    elif ftype == 'hp':
                        foilim = [self.freq_kw[ftype], self.fNy]
                    else:
                        foilim = self.freq_kw[ftype]
                    if direction == 'twopass' and ftype == 'bs':
                        plot_spec(ax, spec, label=f"order {order}", lw=1.5)
                        ax.set_title("Twopass Butterworth bandstop")

        print(spec.show(channel=1, foilim=foilim).sum(), ftype)


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
    #T2 = TestCoherence()
    #T3 = TestCorrelation()
