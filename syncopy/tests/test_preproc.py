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
    time_span = [-.8, 4.2]
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
        spec = freqanalysis(self.data, tapsmofrq=1, keeptrials=False)

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
            spec_f = freqanalysis(filtered, tapsmofrq=1, keeptrials=False)

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
                with pytest.raises(SPYValueError) as err:
                    self.test_but_filter(**kwargs)
                    assert "expected 'onepass'" in str(err)
            else:
                self.test_but_filter(**kwargs)
               
        for order in [-2, 10, 5.6]:
            kwargs = {'direction': 'twopass',
                      'order': order}

            if order < 1 and isinstance(order, int):
                with pytest.raises(SPYValueError) as err:                
                    self.test_but_filter(**kwargs)
                    assert "value to be greater" in str(err)
            elif not isinstance(order, int):
                with pytest.raises(SPYValueError) as err:                
                    self.test_but_filter(**kwargs)
                    assert "expected int_like" in str(err)
            # valid order
            else:
                self.test_but_filter(**kwargs)

    def test_but_selections(self):

        sel_dicts = helpers.mk_selection_dicts(nTrials=20,
                                               nChannels=2,
                                               toi_min=self.time_span[0],
                                               toi_max=self.time_span[1],
                                               min_len=3.5)
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

    def test_but_hilbert_rect(self):

        call = lambda **kwargs: ppfunc(self.data,
                                       freq=20,
                                       filter_class='but',
                                       filter_type='lp',
                                       order=5,
                                       direction='onepass',
                                       **kwargs)

        # test rectification
        filtered = call(rectify=False)
        assert not np.all(filtered.trials[0] > 0)
        rectified = call(rectify=True)
        assert np.all(rectified.trials[0] > 0)

        # test simultaneous call to hilbert and rectification
        with pytest.raises(SPYValueError) as err:        
            call(rectify=True, hilbert='abs')
            assert "either rectifi" in str(err)
            assert "or hilbert" in str(err)

        # test hilbert outputs
        for output in preproc.hilbert_outputs:
            htrafo = call(hilbert=output)
            if output == 'complex':
                assert np.all(np.imag(htrafo.trials[0]) != 0)
            else:
                assert np.all(np.imag(htrafo.trials[0]) == 0)

        # test wrong hilbert parameter
        with pytest.raises(SPYValueError) as err:                
            call(hilbert='absnot')
            assert "one of {'" in str(err)


class TestFIRWS:

    nSamples = 1000
    nChannels = 4
    nTrials = 50
    fs = 200
    fNy = fs / 2

    # -- use flat white noise as test data --

    trls = []
    for _ in range(nTrials):
        trl = np.random.randn(nSamples, nChannels)
        trls.append(trl)

    data = AnalogData(trls, samplerate=fs)
    # for toi tests, -1s offset
    time_span = [-.8, 4.2]
    flow, fhigh = 0.3 * fNy, 0.4 * fNy
    freq_kw = {'lp': fhigh, 'hp': flow,
               'bp': [flow, fhigh], 'bs': [flow, fhigh]}

    def test_firws_filter(self, **kwargs):

        """
        We test for remaining power after filtering
        for all available filter types.
        Order parameter here means length of the filter,
        200 is safe to pass!
        """
        # check if we run the default test
        def_test = not len(kwargs)

        # write default parameters dict
        if def_test:
            kwargs = {'direction': 'twopass',
                      'order': 200}

        # the unfiltered data
        spec = freqanalysis(self.data, tapsmofrq=1, keeptrials=False)
        # total power in arbitrary units (for now)
        pow_tot = spec.show(channel=0).sum()
        nFreq = spec.freq.size

        if def_test:
            fig, ax = mk_spec_ax()

        for ftype in preproc.availableFilterTypes:
            filtered = ppfunc(self.data,
                              filter_class='firws',
                              filter_type=ftype,
                              freq=self.freq_kw[ftype],
                              **kwargs)
            # check in frequency space
            spec_f = freqanalysis(filtered, tapsmofrq=1, keeptrials=False)

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
            ax.set_title(f"Twopass FIRWS, order = {kwargs['order']}")

    def test_firws_kwargs(self):

        """
        Test order and direction parameter
        """

        for direction in preproc.availableDirections:
            kwargs = {'direction': direction,
                      'order': 200}
            self.test_firws_filter(**kwargs)
        for order in [-2, 220, 5.6]:
            kwargs = {'direction': 'twopass',
                      'order': order}

            if order < 1 and isinstance(order, int):
                with pytest.raises(SPYValueError) as err:        
                    self.test_firws_filter(**kwargs)
                    assert "value to be greater" in str(err)

            elif not isinstance(order, int):
                with pytest.raises(SPYValueError) as err:                    
                    self.test_firws_filter(**kwargs)
                    assert "expected int_like" in str(err)

            # valid order
            else:
                self.test_firws_filter(**kwargs)

    def test_firws_selections(self):

        sel_dicts = helpers.mk_selection_dicts(nTrials=20,
                                               nChannels=2,
                                               toi_min=self.time_span[0],
                                               toi_max=self.time_span[1],
                                               min_len=3.5)
        for sd in sel_dicts:
            print(sd)            
            self.test_firws_filter(select=sd, order=200)

    def test_firws_polyremoval(self):

        helpers.run_polyremoval_test(self.test_firws_filter)

    def test_firws_cfg(self):

        cfg = get_defaults(ppfunc)

        cfg.filter_class = 'firws'
        cfg.order = 200
        cfg.direction = 'twopass'
        cfg.freq = 30
        cfg.filter_type = 'hp'

        result = ppfunc(self.data, cfg)

        # check here just for finiteness
        assert np.all(np.isfinite(result.data))

    @skip_without_acme
    def test_firws_parallel(self, testcluster=None):

        ppl.ioff()
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and 'parallel' not in attr)]

        for test_name in all_tests:
            test_method = getattr(self, test_name)
            if 'firws_filter' in test_name:
                # test parallelisation along channels
                test_method(chan_per_worker=2)
            else:
                test_method()
        client.close()
        ppl.ion()

    def test_firws_hilbert_rect(self):

        call = lambda **kwargs: ppfunc(self.data,
                                       freq=20,
                                       filter_class='firws',
                                       filter_type='lp',
                                       order=200,
                                       direction='onepass',
                                       **kwargs)

        # test rectification
        filtered = call(rectify=False)
        assert not np.all(filtered.trials[0] > 0)
        rectified = call(rectify=True)
        assert np.all(rectified.trials[0] > 0)

        # test simultaneous call to hilbert and rectification
        with pytest.raises(SPYValueError) as err:        
            call(rectify=True, hilbert='abs')
            assert "either rectifi" in str(err)
            assert "or hilbert" in str(err)

        # test hilbert outputs
        for output in preproc.hilbert_outputs:
            htrafo = call(hilbert=output)
            if output == 'complex':
                assert np.all(np.imag(htrafo.trials[0]) != 0)
            else:
                assert np.all(np.imag(htrafo.trials[0]) == 0)

        # test wrong hilbert parameter
        with pytest.raises(SPYValueError) as err:        
            call(hilbert='absnot')
            assert "one of {'" in str(err)


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
    T2 = TestFIRWS()

