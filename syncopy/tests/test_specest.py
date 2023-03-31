# -*- coding: utf-8 -*-
#
# Test spectral estimation methods
#

# Builtin/3rd party package imports
import inspect
import random
import psutil
import pytest
import numpy as np
import scipy.signal as scisig
import dask.distributed as dd

# Local imports
from syncopy.tests.misc import generate_artificial_data, flush_local_cluster
from syncopy import freqanalysis, selectdata
from syncopy.shared.errors import SPYValueError, SPYError
from syncopy.datatype.selector import Selector
from syncopy.datatype import AnalogData, SpectralData
from syncopy.shared.tools import StructDict, get_defaults

# Decorator to decide whether or not to run memory-intensive tests
availMem = psutil.virtual_memory().total
skip_low_mem = pytest.mark.skipif(availMem < 10 * 1024**3, reason="less than 10GB RAM available")


# Local helper for constructing TF testing signals
def _make_tf_signal(nChannels, nTrials, seed, fadeIn=None, fadeOut=None, short=False):

    # Construct high-frequency signal modulated by slow oscillating cosine and
    # add time-decaying noise
    nChan2 = int(nChannels / 2)
    fs = 1000
    tStart = -2.95  # FIXME
    tStop = 7.05
    if short:
        fs = 500
        tStart = -0.5  # FIXME
        tStop = 1.5
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    numType = "float32"
    modPeriods = [0.125, 0.0625]
    rng = np.random.default_rng(seed)
    # tStart = -29.5
    # tStop = 70.5
    t0 = -np.abs(tStart * fs).astype(np.intp)
    time = (np.arange(0, (tStop - tStart) * fs, dtype=numType) + tStart * fs) / fs
    N = time.size
    carriers = np.zeros((N, 2), dtype=numType)
    modulators = np.zeros((N, 2), dtype=numType)
    noise_decay = np.exp(-np.arange(N) / (5*fs))
    fader = np.ones((N,), dtype=numType)
    if fadeIn is None:
        fadeIn = tStart
    if fadeOut is None:
        fadeOut = tStop
    fadeIn = np.arange(0, (fadeIn - tStart) * fs, dtype=np.intp)
    fadeOut = np.arange((fadeOut - tStart) * fs, min(10 * fs, N), dtype=np.intp)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    fader[fadeIn] = sigmoid(np.linspace(-2 * np.pi, 2 * np.pi, fadeIn.size))
    fader[fadeOut] = sigmoid(-np.linspace(-2 * np.pi, 2 * np.pi, fadeOut.size))
    for k, period in enumerate(modPeriods):
        modulators[:, k] = 500 * np.cos(2 * np.pi * period * time)
        carriers[:, k] = fader * amp * np.sin(2 * np.pi * 3e2 * time + modulators[:, k])

    # For trials: stitch together carrier + noise, each trial gets its own (fixed
    # but randomized) noise term, channels differ by period in modulator, stratified
    # by trials, i.e.,
    # Trial #0, channels 0, 2, 4, 6, ...: mod -> 0.125 * time
    #                    1, 3, 5, 7, ...: mod -> 0.0625 * time
    # Trial #1, channels 0, 2, 4, 6, ...: mod -> 0.0625 * time
    #                    1, 3, 5, 7, ...: mod -> 0.125 * time
    even = [None, 0, 1]
    odd = [None, 1, 0]
    sig = np.zeros((N * nTrials, nChannels), dtype=numType)
    trialdefinition = np.zeros((nTrials, 3), dtype=np.intp)
    for ntrial in range(nTrials):
        noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape).astype(numType)
        noise *= noise_decay
        nt1 = ntrial * N
        nt2 = (ntrial + 1) * N
        sig[nt1 : nt2, ::2] = np.tile(carriers[:, even[(-1)**ntrial]] + noise, (nChan2, 1)).T
        sig[nt1 : nt2, 1::2] = np.tile(carriers[:, odd[(-1)**ntrial]] + noise, (nChan2, 1)).T
        trialdefinition[ntrial, :] = np.array([nt1, nt2, t0])

    # Finally allocate `AnalogData` object that makes use of all this
    tfData = AnalogData(data=sig, samplerate=fs, trialdefinition=trialdefinition)

    return tfData, modulators, even, odd, fader


class TestMTMFFT():


    # Construct simple trigonometric signal to check FFT consistency: each
    # channel is a sine wave of frequency `freqs[nchan]` with single unique
    # amplitude `amp` and sampling frequency `fs`
    nChannels = 32
    nTrials = 8
    fs = 1024
    fband = np.linspace(1, fs/2, int(np.floor(fs/2)))
    freqs = [88.,  35., 278., 104., 405., 314., 271., 441., 343., 374., 428.,
             367., 75., 118., 289., 310., 510., 102., 123., 417., 273., 449.,
             416., 32., 438., 111., 140., 304., 327., 494., 23., 493.]
    freqs = freqs[:nChannels]
    # freqs = np.random.choice(fband[:-2], size=nChannels, replace=False)
    amp = np.pi
    phases = np.random.permutation(np.linspace(0, 2 * np.pi, nChannels))
    t = np.linspace(0, nTrials, nTrials * fs)
    sig = np.zeros((t.size, nChannels), dtype="float32")
    for nchan in range(nChannels):
        sig[:, nchan] = amp * \
            np.sin(2 * np.pi * freqs[nchan] * t + phases[nchan])

    trialdefinition = np.zeros((nTrials, 3), dtype="int")
    for ntrial in range(nTrials):
        trialdefinition[ntrial, :] = np.array(
            [ntrial * fs, (ntrial + 1) * fs, 0])

    adata = AnalogData(data=sig, samplerate=fs,
                       trialdefinition=trialdefinition)

    # Data selections to be tested w/data generated based on `sig`
    sigdataSelections = [None,
                         {"trials": [3, 1, 0],
                          "channel": ["channel" + str(i) for i in range(12, 28)][::-1]},
                         {"trials": [0, 1, 2],
                          "channel": range(0, int(nChannels / 2)),
                          "latency": [0.25, 0.75]}]

    # Data selections to be tested w/`artdata` generated below (use fixed but arbitrary
    # random number seed to randomly select time-points for `toi` (with repetitions)
    seed = np.random.RandomState(13)
    artdataSelections = [None,
                         {"trials": [3, 1, 0],
                          "channel": ["channel" + str(i) for i in range(10, 15)][::-1]},
                         {"trials": [0, 1, 2],
                          "channel": range(0, 8),
                          "latency": [-0.5, 0.6]}]

    # Error tolerances for target amplitudes (depend on data selection!)
    tols = [1, 1, 1.5]

    # Error tolerance for frequency-matching
    ftol = 0.25

    # Helper function that reduces dataselections (keep `None` selection no matter what)
    def test_cut_selections(self):
        self.sigdataSelections.pop(random.choice([-1, 1]))
        self.artdataSelections.pop(random.choice([-1, 1]))

    @staticmethod
    def get_adata():
        return AnalogData(data=TestMTMFFT.sig, samplerate=TestMTMFFT.fs,
                       trialdefinition=TestMTMFFT.trialdefinition)

    def test_output(self):
        # ensure that output type specification is respected
        for select in self.sigdataSelections:
            spec = freqanalysis(self.adata, method="mtmfft", taper="hann",
                                output="fourier", select=select)
            assert "complex" in spec.data.dtype.name
            spec = freqanalysis(self.adata, method="mtmfft", taper="hann",
                                output="abs", select=select)
            assert "float" in spec.data.dtype.name
            spec = freqanalysis(self.adata, method="mtmfft", taper="hann",
                                output="pow", select=select)
            assert "float" in spec.data.dtype.name

    def test_solution(self):
        # ensure channel-specific frequencies are identified correctly
        for sk, select in enumerate(self.sigdataSelections):
            sel = Selector(self.adata, select)
            spec = freqanalysis(self.adata, method="mtmfft", taper="hann",
                                pad="nextpow2", output="pow", select=select)

            chanList = np.arange(self.nChannels)[sel.channel]
            amps = np.empty((len(sel.trial_ids) * len(chanList),))
            k = 0
            for nchan, chan in enumerate(chanList):
                for ntrial in range(len(spec.trials)):
                    amps[k] = spec.data[ntrial, :, :, nchan].max() / \
                        self.t.size
                    assert np.argmax(
                            spec.data[ntrial, :, :, nchan]) == self.freqs[chan]
                    k += 1

            # ensure amplitude is consistent across all channels/trials
            assert np.all(np.diff(amps) < self.tols[sk])

    def test_normalization(self):

        nSamples = 1000
        fsample = 500  # 2s long signal
        Ampl = 4  # amplitude
        # 50Hz harmonic, spectral power is given by: Ampl^2 / 2 = 8
        signal = Ampl * np.cos(2 * np.pi * 50 * np.arange(nSamples) * 1 / fsample)

        # single signal/channel is enough
        ad = AnalogData([signal[:, None]], samplerate=fsample)

        cfg = StructDict()
        cfg.foilim = [40, 60]
        cfg.output = 'pow'
        cfg.taper = None

        # -- syncopy's default, padding does NOT change power --

        cfg.ft_compat = False
        cfg.pad = 'maxperlen'  # that's the default -> no padding
        spec = freqanalysis(ad, cfg)
        peak_power = spec.show().max()
        df_no_pad = np.diff(spec.freq)  # freq. resolution
        assert np.allclose(peak_power, Ampl**2 / 2, atol=1e-5)

        cfg.pad = 4  # in seconds, double the size
        spec = freqanalysis(ad, cfg)
        df_with_pad = np.diff(spec.freq)
        # we double the spectral resolution
        assert np.allclose(df_no_pad[0], 2 * df_with_pad[0])
        # yet power stays the same
        peak_power = spec.show().max()
        assert np.allclose(peak_power, Ampl**2 / 2, atol=1e-5)

        # -- FT compat mode, padding does dilute the power --

        cfg.ft_compat = True
        cfg.pad = 'maxperlen'  # that's the default
        spec = freqanalysis(ad, cfg)
        peak_power = spec.show().max()
        df_no_pad = np.diff(spec.freq)
        # default padding is no padding if all trials are equally sized,
        # so here the results are the same
        assert np.allclose(peak_power, Ampl**2 / 2, atol=1e-5)

        cfg.pad = 4  # in seconds, double the size
        spec = freqanalysis(ad, cfg)
        df_with_pad = np.diff(spec.freq)
        # we double the spectral resolution
        assert np.allclose(df_no_pad[0], df_with_pad[0] * 2)
        # here half the power is now lost!
        peak_power = spec.show().max()
        assert np.allclose(peak_power, Ampl**2 / 4, atol=1e-5)

        # -- works the same with tapering --

        cfg.ft_compat = False
        cfg.pad = 'maxperlen'  # that's the default
        cfg.taper = 'kaiser'
        cfg.taper_opt = {'beta': 10}
        spec = freqanalysis(ad, cfg)
        peak_power_no_pad = spec.show().max()

        cfg.pad = 4
        spec = freqanalysis(ad, cfg)
        peak_power_with_pad = spec.show().max()
        assert np.allclose(peak_power_no_pad, peak_power_with_pad, atol=1e-5)

        cfg.ft_compat = True
        cfg.pad = 'maxperlen'  # that's the default
        cfg.taper = 'kaiser'
        cfg.taper_opt = {'beta': 10}
        spec = freqanalysis(ad, cfg)
        peak_power_no_pad = spec.show().max()

        cfg.pad = 4
        spec = freqanalysis(ad, cfg)
        peak_power_with_pad = spec.show().max()
        # again half the power is lost with FT compat
        assert np.allclose(peak_power_no_pad, 2 * peak_power_with_pad, atol=1e-5)

    def test_foi(self):
        for select in self.sigdataSelections:

            # `foi` lims outside valid bounds
            with pytest.raises(SPYValueError):
                freqanalysis(TestMTMFFT.get_adata(), method="mtmfft", taper="hann",
                             foi=[-0.5, self.fs / 3], select=select)
            with pytest.raises(SPYValueError):
                freqanalysis(TestMTMFFT.get_adata(), method="mtmfft", taper="hann",
                             foi=[1, self.fs], select=select)

            foi = self.fband[1:int(self.fband.size / 3)]

            # offset `foi` by 0.1 Hz - resulting freqs must be unaffected
            ftmp = foi + 0.1
            spec = freqanalysis(TestMTMFFT.get_adata(), method="mtmfft", taper="hann",
                                pad="nextpow2", foi=ftmp, select=select)
            assert np.all(spec.freq == foi)

            # unsorted, duplicate entries in `foi` - result must stay the same
            ftmp = np.hstack([foi, np.full(20, foi[0])])
            spec = freqanalysis(TestMTMFFT.get_adata(), method="mtmfft", taper="hann",
                                pad="nextpow2", foi=ftmp, select=select)
            assert np.all(spec.freq == foi)

    def test_dpss(self):

        for select in self.sigdataSelections:

            self.adata.selectdata(select, inplace=True)
            sel = self.adata.selection
            chanList = np.arange(self.nChannels)[sel.channel]
            self.adata.selection = None

            # ensure default setting results in single taper
            spec = freqanalysis(self.adata, method="mtmfft",
                                tapsmofrq=3, output="pow", select=select)
            assert spec.taper.size == 1
            assert spec.channel.size == len(chanList)

            # specify tapers
            spec = freqanalysis(self.adata, method="mtmfft",
                                tapsmofrq=7, keeptapers=True, select=select)
            assert spec.channel.size == len(chanList)

            # trigger capture of too large tapsmofrq (edge case)
            spec = freqanalysis(self.adata, method="mtmfft",
                                tapsmofrq=2, output="pow", select=select)

        # non-equidistant data w/multiple tapers
        artdata = generate_artificial_data(nTrials=5, nChannels=16,
                                           equidistant=False, inmemory=False)
        timeAxis = artdata.dimord.index("time")
        cfg = StructDict()
        cfg.method = "mtmfft"
        cfg.tapsmofrq = 3.3
        cfg.output = "pow"

        for select in self.artdataSelections:

            sel = selectdata(artdata, select)
            cfg.select = select
            spec = freqanalysis(cfg, artdata)

            # ensure correctness of padding (respecting min. trial length + time-selection)
            if select is None:
                maxtrlno = np.diff(artdata.sampleinfo).argmax()
                nSamples = artdata.trials[maxtrlno].shape[timeAxis]
            else:
                nSamples = max([trl.shape[0] for trl in sel.trials])
            freqs = np.fft.rfftfreq(nSamples, 1 / artdata.samplerate)
            assert spec.freq.size == freqs.size
            assert np.max(spec.freq - freqs) < self.ftol

        # same + reversed dimensional order in input object
        cfg.data = generate_artificial_data(nTrials=5, nChannels=16,
                                            equidistant=False, inmemory=False,
                                            dimord=AnalogData._defaultDimord[::-1])
        timeAxis = cfg.data.dimord.index("time")
        cfg.output = "abs"
        cfg.keeptapers = True

        for select in self.artdataSelections:
            sel = selectdata(cfg.data, select)
            cfg.select = select

            spec = freqanalysis(cfg)

            # ensure correctness of padding (respecting min. trial length + time-selection)
            if select is None:
                maxtrlno = np.diff(cfg.data.sampleinfo).argmax()
                nSamples = cfg.data.trials[maxtrlno].shape[timeAxis]
            else:
                nSamples = max([trl.shape[1] for trl in sel.trials])

            freqs = np.fft.rfftfreq(nSamples, 1 / cfg.data.samplerate)
            assert spec.freq.size == freqs.size
            assert np.max(spec.freq - freqs) < self.ftol
            assert spec.taper.size > 1

        # same + overlapping trials
        cfg.data = generate_artificial_data(nTrials=5, nChannels=16,
                                            equidistant=False, inmemory=False,
                                            dimord=AnalogData._defaultDimord[::-1],
                                            overlapping=True)
        timeAxis = cfg.data.dimord.index("time")
        cfg.keeptapers = False
        cfg.output = "pow"

        for select in self.artdataSelections:

            sel = selectdata(cfg.data, select)
            cfg.select = select

            spec = freqanalysis(cfg)

            # ensure correctness of padding (respecting min. trial length + time-selection)
            if select is None:
                maxtrlno = np.diff(cfg.data.sampleinfo).argmax()
                nSamples = cfg.data.trials[maxtrlno].shape[timeAxis]
            else:
                nSamples = max([trl.shape[timeAxis] for trl in sel.trials])

            freqs = np.fft.rfftfreq(nSamples, 1 / cfg.data.samplerate)
            assert spec.freq.size == freqs.size
            assert np.max(spec.freq - freqs) < self.ftol
            assert spec.taper.size == 1

    @skip_low_mem
    def test_parallel(self, testcluster):
        # collect all tests of current class and repeat them using dask
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and attr not in ["test_parallel", "test_cut_selections"])]
        for test in all_tests:
            getattr(self, test)()
            flush_local_cluster(testcluster)

        # now create uniform `cfg` for remaining SLURM tests
        cfg = StructDict()
        cfg.method = "mtmfft"
        cfg.tapsmofrq = 9.3
        cfg.output = "pow"

        # no. of HDF5 files that will make up virtual data-set in case of channel-chunking
        chanPerWrkr = 7
        nFiles = self.nTrials * (int(self.nChannels / chanPerWrkr)
                                 + int(self.nChannels % chanPerWrkr > 0))

        # simplest case: equidistant trial spacing, all in memory
        fileCount = [self.nTrials, nFiles]
        artdata = generate_artificial_data(nTrials=self.nTrials, nChannels=self.nChannels,
                                           inmemory=True)
        for k, chan_per_worker in enumerate([None, chanPerWrkr]):
            cfg.chan_per_worker = chan_per_worker
            spec = freqanalysis(artdata, cfg)
            assert spec.data.is_virtual
            assert len(spec.data.virtual_sources()) == fileCount[k]

        # non-equidistant trial spacing
        cfg.keeptapers = False
        artdata = generate_artificial_data(nTrials=self.nTrials, nChannels=self.nChannels,
                                           inmemory=True, equidistant=False)
        timeAxis = artdata.dimord.index("time")
        maxtrlno = np.diff(artdata.sampleinfo).argmax()
        nSamples = artdata.trials[maxtrlno].shape[timeAxis]
        freqs = np.fft.rfftfreq(nSamples, 1 / artdata.samplerate)
        for k, chan_per_worker in enumerate([None, chanPerWrkr]):
            spec = freqanalysis(artdata, cfg)
            assert spec.freq.size == freqs.size
            assert np.allclose(spec.freq, freqs)
            assert spec.taper.size == 1

        # equidistant trial spacing, keep tapers
        cfg.output = "abs"
        cfg.keeptapers = True
        artdata = generate_artificial_data(nTrials=self.nTrials, nChannels=self.nChannels,
                                           inmemory=False)
        for k, chan_per_worker in enumerate([None, chanPerWrkr]):
            spec = freqanalysis(artdata, cfg)
            assert spec.taper.size > 1

        # non-equidistant, overlapping trial spacing, throw away trials and tapers
        cfg.keeptapers = False
        cfg.keeptrials = "no"
        cfg.output = "pow"
        artdata = generate_artificial_data(nTrials=self.nTrials, nChannels=self.nChannels,
                                           inmemory=False, equidistant=False,
                                           overlapping=True)
        spec = freqanalysis(artdata, cfg)
        timeAxis = artdata.dimord.index("time")
        maxtrlno = np.diff(artdata.sampleinfo).argmax()
        nSamples = artdata.trials[maxtrlno].shape[timeAxis]
        freqs = np.fft.rfftfreq(nSamples, 1 / artdata.samplerate)
        assert spec.freq.size == freqs.size
        assert np.allclose(spec.freq, freqs)
        assert spec.taper.size == 1
        assert len(spec.time) == 1
        assert len(spec.time[0]) == 1
        assert spec.data.shape == (1, 1, freqs.size, self.nChannels)

        client.close()

    # FIXME: check polyremoval once supported


class TestMTMConvol():

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

    @staticmethod
    def get_tfdata_mtmconvol():
        """
        High-frequency signal modulated by slow oscillating cosine and time-decaying noise.
        """
        return _make_tf_signal(TestMTMConvol.nChannels, TestMTMConvol.nTrials, TestMTMConvol.seed,
                                                           fadeIn=TestMTMConvol.fadeIn, fadeOut=TestMTMConvol.fadeOut)[0]


    # Data selection dict for the above object
    dataSelections = [None,
                      {"trials": [1, 2, 0],
                       "channel": ["channel" + str(i) for i in range(2, 6)][::-1]},
                      {"trials": [0, 2],
                       "channel": range(0, nChan2),
                       "latency": [-2, 6.8]}]
                    #    "latency": [-20, 60.8]}] FIXME

    # Helper function that reduces dataselections (keep `None` selection no matter what)
    def test_tf_cut_selections(self):
        self.dataSelections.pop(random.choice([-1, 1]))

    def test_tf_output(self):
        # Set up basic TF analysis parameters to not slow down things too much
        cfg = get_defaults(freqanalysis)
        cfg.method = "mtmconvol"
        cfg.taper = "hann"
        cfg.toi = np.linspace(-2, 6, 10)
        cfg.t_ftimwin = 1.0
        outputDict = {"fourier": "complex", "abs": "float", "pow": "float"}

        for select in self.dataSelections:
            cfg.select = select
            if select is not None and "latency" in cfg.select.keys():
                with pytest.raises(SPYValueError):
                    freqanalysis(cfg, self.tfData)

                self.tfData.selection = None
                continue

            for key, value in outputDict.items():
                cfg.output = key
                tfSpec = freqanalysis(cfg, self.tfData)
                assert value in tfSpec.data.dtype.name

    def test_tf_solution(self):
        # Compute "full" non-overlapping TF spectrum, i.e., center analysis windows
        # on all time-points with window-boundaries touching but not intersecting
        cfg = get_defaults(freqanalysis)
        cfg.method = "mtmconvol"
        cfg.taper = "hann"
        cfg.t_ftimwin = 1.0
        cfg.toi = 0
        cfg.output = "pow"
        cfg.keeptapers = False

        # Set up index tuple for slicing computed TF spectra and collect values
        # of expected frequency peaks (for validation of `foi`/`foilim` selections below)
        chanIdx = SpectralData._defaultDimord.index("channel")
        tfIdx = [slice(None)] * len(SpectralData._defaultDimord)
        maxFreqs = np.hstack([np.arange(325, 336), np.arange(355,366)])
        allFreqs = np.arange(self.tfData.samplerate / 2 + 1)
        foilimFreqs = np.arange(maxFreqs.min(), maxFreqs.max() + 1)

        for select in self.dataSelections:

            # Compute TF objects w\w/o`foi`/`foilim`
            cfg.select = select
            tfSpec = freqanalysis(cfg, TestMTMConvol.get_tfdata_mtmconvol())
            cfg.foi = maxFreqs
            tfSpecFoi = freqanalysis(cfg, TestMTMConvol.get_tfdata_mtmconvol())
            cfg.foi = None
            cfg.foilim = [maxFreqs.min(), maxFreqs.max()]
            tfSpecFoiLim = freqanalysis(cfg, TestMTMConvol.get_tfdata_mtmconvol())
            cfg.foilim = None

            # Ensure TF objects contain expected/requested frequencies
            assert np.array_equal(tfSpec.freq, allFreqs)
            assert np.array_equal(tfSpecFoi.freq, maxFreqs)
            assert np.array_equal(tfSpecFoiLim.freq, foilimFreqs)

            for tk, trlArr in enumerate(tfSpec.trials):
                tfData = TestMTMConvol.get_tfdata_mtmconvol()

                # Compute expected timing array depending on `latency`
                trlNo = tk
                timeArr = np.arange(tfData.time[trlNo][0], tfData.time[trlNo][-1])
                timeSelection = slice(None)
                if select:
                    trlNo = select["trials"][tk]
                    if "latency" in select.keys():
                        timeArr = np.arange(*select["latency"])
                        timeStart = int(select['latency'][0] * tfData.samplerate - tfData._t0[trlNo])
                        timeStop = int(select['latency'][1] * tfData.samplerate - tfData._t0[trlNo])
                        timeSelection = slice(timeStart, timeStop)

                # Ensure timing array was computed correctly and independent of `foi`/`foilim`
                assert np.array_equal(timeArr, tfSpec.time[tk])
                assert np.array_equal(tfSpec.time[tk], tfSpecFoi.time[tk])
                assert np.array_equal(tfSpecFoi.time[tk], tfSpecFoiLim.time[tk])

                for chan in range(tfSpec.channel.size):

                    # Get reference channel in input object to determine underlying modulator
                    chanNo = chan
                    if select:
                        if "latency" not in select.keys():
                            chanNo = np.where(tfData.channel == select["channel"][chan])[0][0]
                    if chanNo % 2:
                        modIdx = self.odd[(-1)**trlNo]
                    else:
                        modIdx = self.even[(-1)**trlNo]
                    tfIdx[chanIdx] = chan
                    Zxx = trlArr[tuple(tfIdx)].squeeze()

                    # Use SciPy's `find_peaks` to identify frequency peaks in computed TF spectrum:
                    # `peakProfile` is just a sliver of the TF spectrum around the peak frequency; to
                    # better understand what's happening here, look at
                    # plt.figure(); plt.plot(peakProfile); plt.plot(peaks, peakProfile[peaks], 'x')
                    ZxxMax = Zxx.max()
                    ZxxThresh = 0.1 * ZxxMax
                    _, freqPeaks = np.where(Zxx >= (ZxxMax - ZxxThresh))
                    freqMax, freqMin = freqPeaks.max(), freqPeaks.min()
                    modulator = self.modulators[timeSelection, modIdx]
                    modCounts = [sum(modulator == modulator.min()), sum(modulator == modulator.max())]
                    for fk, freqPeak in enumerate([freqMin, freqMax]):
                        peakProfile = Zxx[:, freqPeak - 1 : freqPeak + 2].mean(axis=1)
                        peaks, _ = scisig.find_peaks(peakProfile, height=ZxxThresh)
                        assert np.abs(peaks.size - modCounts[fk]) <= 1

                    # Ensure that the `foi`/`foilim` selections correspond to the respective
                    # slivers of the full TF spectrum
                    assert np.allclose(tfSpecFoi.trials[tk][tuple(tfIdx)].squeeze(),
                                       Zxx[:, maxFreqs])
                    assert np.allclose(tfSpecFoiLim.trials[tk][tuple(tfIdx)].squeeze(),
                                       Zxx[:, maxFreqs.min():maxFreqs.max() + 1])

    def test_tf_toi(self):
        # Use a Hanning window and throw away trials to speed up things a bit
        cfg = get_defaults(freqanalysis)
        cfg.method = "mtmconvol"
        cfg.taper = "hann"
        cfg.output = "pow"
        cfg.keeptrials = False
        cfg.keeptapers = False

        # Test various combinations of `toi` and `t_ftimwin`: `toiArrs` comprises
        # arrays containing the onset, purely pre-onset, purely after onset and
        # non-unit spacing
        toiVals = [0.9, 0.75]
        toiArrs = [np.arange(-2, 7),
                   np.arange(-1, 6, 1 / self.tfData.samplerate),
                   np.arange(1, 6, 2)]
        winSizes = [0.5, 1.0]

        # Combine `toi`-testing w/in-place data-pre-selection
        for select in self.dataSelections:
            cfg.select = select
            tStart = self.tfData.time[0][0]
            tStop = self.tfData.time[0][-1]
            if select:
                if "latency" in select.keys():
                    tStart = select["latency"][0]
                    tStop = select["latency"][1]

            # Test TF calculation w/different window-size/-centroids: ensure
            # resulting timing arrays are correct
            dt = TestMTMConvol.get_tfdata_mtmconvol()
            for winsize in winSizes:
                cfg.t_ftimwin = winsize
                for toi in toiVals:
                    cfg.toi = toi
                    tfSpec = freqanalysis(cfg, dt)
                    tStep = winsize - toi * winsize
                    timeArr = np.arange(tStart, tStop, tStep)
                    assert np.allclose(timeArr, tfSpec.time[0])

            # Test window-centroids specified as time-point arrays
            dt = TestMTMConvol.get_tfdata_mtmconvol()
            if select is not None and "latency" not in select.keys():
                cfg.t_ftimwin = 0.05
                for toi in toiArrs:
                    cfg.toi = toi
                    tfSpec = freqanalysis(cfg, dt)
                    assert np.allclose(cfg.toi, tfSpec.time[0])
                    assert tfSpec.samplerate == 1/(toi[1] - toi[0])

                # Unevenly sampled array: timing currently in lala-land, but sizes must match
                cfg.toi = [-1, 2, 6]
                tfSpec = freqanalysis(cfg, TestMTMConvol.get_tfdata_mtmconvol())
                assert tfSpec.time[0].size == len(cfg.toi)

        # Test correct time-array assembly for ``toi = "all"`` (cut down data signifcantly
        # to not overflow memory here); same for ``toi = 1.0```
        cfg.tapsmofrq = 10
        cfg.keeptapers = True
        cfg.select = {"trials": [0], "channel": [0], "latency": [-0.5, 0.5]}
        cfg.toi = "all"
        cfg.t_ftimwin = 0.05
        tfSpec = freqanalysis(cfg, TestMTMConvol.get_tfdata_mtmconvol())
        assert tfSpec.taper.size >= 1
        dt = 1 / self.tfData.samplerate
        timeArr = np.arange(cfg.select["latency"][0], cfg.select["latency"][1] + dt, dt)
        assert np.allclose(tfSpec.time[0], timeArr)
        cfg.toi = 1.0
        tfSpec = freqanalysis(cfg, TestMTMConvol.get_tfdata_mtmconvol())
        assert np.allclose(tfSpec.time[0], timeArr)

        # Use a window-size larger than the pre-selected interval defined above
        cfg.t_ftimwin = 5.0
        with pytest.raises(SPYValueError) as spyval:
            freqanalysis(cfg, TestMTMConvol.get_tfdata_mtmconvol())
            assert "Invalid value of `t_ftimwin`" in str(spyval.value)
        cfg.t_ftimwin = 0.05

        # Use `toi` array outside trial boundaries
        cfg.toi = self.tfData.time[0][:10]
        with pytest.raises(SPYError) as spyval:
            freqanalysis(cfg, TestSuperlet._get_tf_data_superlet())
            errmsg = "Invalid value of `toi`: expected all array elements to be bounded by {} and {}"
            assert errmsg.format(*cfg.select["latency"]) in str(spyval.value)

        # Unsorted `toi` array
        cfg.toi = [0.3, -0.1, 0.2]
        with pytest.raises(SPYError) as spyval:
            freqanalysis(cfg, TestMTMConvol.get_tfdata_mtmconvol())

    def test_tf_irregular_trials(self):
        # Settings for computing "full" non-overlapping TF-spectrum with DPSS tapers:
        # ensure non-equidistant/overlapping trials are processed (padded) correctly
        # also make sure ``toi = "all"`` works under any circumstance
        cfg = get_defaults(freqanalysis)
        cfg.method = "mtmconvol"
        cfg.tapsmofrq = 2
        cfg.t_ftimwin = 0.3
        cfg.output = "pow"
        cfg.keeptapers = True

        nTrials = 2
        nChannels = 2

        # start harmless: equidistant trials w/multiple tapers
        cfg.toi = 0.0
        # this guy always creates a data set from [-1, ..., 1.9999] seconds
        # no way to change this..
        artdata_len = 3
        artdata = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                           equidistant=True, inmemory=False)
        tfSpec = freqanalysis(artdata, **cfg)
        assert tfSpec.taper.size >= 1
        for trl_time in tfSpec.time:
            assert np.allclose(artdata_len / cfg.t_ftimwin, trl_time[0].shape)

        cfg.toi = "all"
        artdata = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                           equidistant=True, inmemory=False)
        # reduce samples, otherwise the the memory usage explodes (nSamples x win_size x nFreq)
        rdat = artdata.selectdata(latency=[0, 0.5])
        tfSpec = freqanalysis(rdat, **cfg)
        for tk, origTime in enumerate(rdat.time):
            assert np.array_equal(origTime, tfSpec.time[tk])

        # non-equidistant trials w/multiple tapers
        cfg.toi = 0.0
        artdata = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                           equidistant=False, inmemory=False)
        tfSpec = freqanalysis(artdata, **cfg)
        assert tfSpec.taper.size >= 1
        for tk, trl_time in enumerate(tfSpec.time):
            assert np.allclose(np.ceil(artdata.time[tk].size / artdata.samplerate / cfg.t_ftimwin), trl_time.size)

        cfg.toi = "all"
        # reduce samples, otherwise the the memory usage explodes (nSamples x win_size x nFreq)
        rdat = artdata.selectdata(latency=[0, 0.5])
        tfSpec = freqanalysis(rdat, **cfg)
        for tk, origTime in enumerate(rdat.time):
            assert np.array_equal(origTime, tfSpec.time[tk])

        # same + reversed dimensional order in input object
        cfg.toi = 0.0
        artdata = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                           equidistant=False, inmemory=False,
                                           dimord=AnalogData._defaultDimord[::-1])
        tfSpec = freqanalysis(artdata, cfg)
        assert tfSpec.taper.size >= 1
        for tk, trl_time in enumerate(tfSpec.time):
            assert np.allclose(np.ceil(artdata.time[tk].size / artdata.samplerate / cfg.t_ftimwin), trl_time.size)

        cfg.toi = "all"
        # reduce samples, otherwise the the memory usage explodes (nSamples x win_size x nFreq)
        rdat = artdata.selectdata(latency=[0, 0.5])
        tfSpec = freqanalysis(rdat, cfg)

        # same + overlapping trials
        cfg.toi = 0.0
        artdata = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                           equidistant=False, inmemory=False,
                                           dimord=AnalogData._defaultDimord[::-1],
                                           overlapping=True)
        tfSpec = freqanalysis(artdata, cfg)
        assert tfSpec.taper.size >= 1
        for tk, trl_time in enumerate(tfSpec.time):
            assert np.allclose(np.ceil(artdata.time[tk].size / artdata.samplerate / cfg.t_ftimwin), trl_time.size)

    @skip_low_mem
    def test_tf_parallel(self, testcluster):
        # collect all tests of current class and repeat them running concurrently
        client = dd.Client(testcluster)
        quick_tests = [attr for attr in self.__dir__()
                       if (inspect.ismethod(getattr(self, attr)) and attr not in ["test_tf_parallel", "test_tf_cut_selections"])]
        slow_tests = []
        slow_tests.append(quick_tests.pop(quick_tests.index("test_tf_output")))
        slow_tests.append(quick_tests.pop(quick_tests.index("test_tf_irregular_trials")))
        for test in quick_tests:
            getattr(self, test)()
            flush_local_cluster(testcluster)
        for test in slow_tests:
            getattr(self, test)()
            flush_local_cluster(testcluster)

        # now create uniform `cfg` for remaining SLURM tests
        cfg = StructDict()
        cfg.method = "mtmconvol"
        cfg.taper = "hann"
        cfg.t_ftimwin = 1.0
        cfg.toi = 0
        cfg.output = "pow"

        nChannels = 3
        nTrials = 2

        # no. of HDF5 files that will make up virtual data-set in case of channel-chunking
        chanPerWrkr = 2
        nFiles = nTrials * (int(nChannels/chanPerWrkr)
                            + int(nChannels % chanPerWrkr > 0))

        # simplest case: equidistant trial spacing, all in memory
        fileCount = [nTrials, nFiles]
        artdata = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                           inmemory=True)
        for k, chan_per_worker in enumerate([None, chanPerWrkr]):
            cfg.chan_per_worker = chan_per_worker
            tfSpec = freqanalysis(artdata, cfg)
            assert tfSpec.data.is_virtual
            assert len(tfSpec.data.virtual_sources()) == fileCount[k]

        # non-equidistant trial spacing
        cfg.keeptapers = False
        artdata = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                           inmemory=True, equidistant=False)
        expectedFreqs = np.arange(artdata.samplerate / 2 + 1)
        for chan_per_worker in enumerate([None, chanPerWrkr]):
            tfSpec = freqanalysis(artdata, cfg)
            assert np.array_equal(tfSpec.freq, expectedFreqs)
            assert tfSpec.taper.size == 1

        # equidistant trial spacing, keep tapers
        cfg.output = "abs"
        cfg.tapsmofrq = 2
        cfg.keeptapers = True
        artdata = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                           inmemory=False)
        for chan_per_worker in enumerate([None, chanPerWrkr]):
            tfSpec = freqanalysis(artdata, cfg)
            assert tfSpec.taper.size >= 1

        # overlapping trial spacing, throw away trials and tapers
        cfg.keeptapers = False
        cfg.keeptrials = "no"
        cfg.output = "pow"
        artdata = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                           inmemory=False, equidistant=True,
                                           overlapping=True)
        expectedFreqs = np.arange(artdata.samplerate / 2 + 1)
        tfSpec = freqanalysis(artdata, cfg)
        assert np.array_equal(tfSpec.freq, expectedFreqs)
        assert tfSpec.taper.size == 1
        assert np.array_equal(np.unique(np.floor(artdata.time[0])), tfSpec.time[0])
        assert tfSpec.data.shape == (tfSpec.time[0].size, 1, expectedFreqs.size, nChannels)

        client.close()


class TestWavelet():

    # Prepare testing signal: ensure `fadeIn` and `fadeOut` are compatible w/`latency`
    # selection below
    nChannels = 4
    nTrials = 3
    seed = 151120
    fadeIn = -1.5
    fadeOut = 5.5
    tfData, modulators, even, odd, fader = _make_tf_signal(nChannels, nTrials, seed,
                                                           fadeIn=fadeIn, fadeOut=fadeOut)

    @staticmethod
    def get_tfdata_wavelet():
        return(_make_tf_signal(TestWavelet.nChannels, TestWavelet.nTrials, TestWavelet.seed,
                                                           fadeIn=TestWavelet.fadeIn, fadeOut=TestWavelet.fadeOut)[0])

    # Set up in-place data-selection dicts for the constructed object
    dataSelections = [None,
                      {"trials": [1, 2, 0],
                       "channel": ["channel" + str(i) for i in range(2, 4)][::-1]},
                      {"trials": [0, 2],
                       "channel": range(0, int(nChannels / 2)),
                       "latency": [-2, 6.8]}]

    # Helper function that reduces dataselections (keep `None` selection no matter what)
    def test_wav_cut_selections(self):
        self.dataSelections.pop(random.choice([-1, 1]))

    @skip_low_mem
    def test_wav_solution(self):

        # Compute TF specturm across entire time-interval (use integer-valued
        # time-points as wavelet centroids)
        cfg = get_defaults(freqanalysis)
        cfg.method = "wavelet"
        cfg.wavelet = "Morlet"
        cfg.width = 1
        cfg.output = "pow"

        # import pdb; pdb.set_trace()


        # Set up index tuple for slicing computed TF spectra and collect values
        # of expected frequency peaks (for validation of `foi`/`foilim` selections below)
        chanIdx = SpectralData._defaultDimord.index("channel")
        tfIdx = [slice(None)] * len(SpectralData._defaultDimord)
        modFreqs = [330, 360]
        maxFreqs = np.hstack([np.arange(modFreqs[0] - 5, modFreqs[0] + 6),
                              np.arange(modFreqs[1] - 5, modFreqs[1] + 6)])
        foilimFreqs = np.arange(maxFreqs.min(), maxFreqs.max() + 1)

        for select in self.dataSelections:

            # Timing of `tfData` is identical for all trials, so to speed things up,
            # set up `timeArr` here - if `tfData` is modified, these computations have
            # to be moved inside the `enumerate(tfSpec.trials)`-loop!
            timeArr = np.arange(self.tfData.time[0][0], self.tfData.time[0][-1])
            if select:
                if "latency" in select.keys():
                    continue
                    timeArr = np.arange(*select["latency"])
                    timeStart = int(select['latency'][0] * self.tfData.samplerate - self.tfData._t0[0])
                    timeStop = int(select['latency'][1] * self.tfData.samplerate - self.tfData._t0[0])
                    timeSelection = slice(timeStart, timeStop)
            else:
                timeSelection = np.where(self.fader == 1.0)[0]
            cfg.toi = timeArr

            # Compute TF objects w\w/o`foi`/`foilim`
            cfg.select = select
            tfSpec = freqanalysis(cfg, TestSuperlet._get_tf_data_superlet())
            cfg.foi = maxFreqs
            tfSpecFoi = freqanalysis(cfg, TestWavelet.get_tfdata_wavelet())
            cfg.foi = None
            cfg.foilim = [maxFreqs.min(), maxFreqs.max()]
            tfSpecFoiLim = freqanalysis(cfg, TestWavelet.get_tfdata_wavelet())
            cfg.foilim = None

            # Ensure TF objects contain expected/requested frequencies
            assert 0.2 > tfSpec.freq.min() > 0
            assert tfSpec.freq.max() == (self.tfData.samplerate / 2)
            assert tfSpec.freq.size > 40
            assert np.allclose(tfSpecFoi.freq, maxFreqs)
            assert np.allclose(tfSpecFoiLim.freq, foilimFreqs)

            for tk, trlArr in enumerate(tfSpec.trials):

                # Get reference trial-number in input object
                trlNo = tk
                if select:
                    trlNo = select["trials"][tk]

                # Ensure timing array was computed correctly and independent of `foi`/`foilim`
                assert np.array_equal(timeArr, tfSpec.time[tk])
                assert np.array_equal(tfSpec.time[tk], tfSpecFoi.time[tk])
                assert np.array_equal(tfSpecFoi.time[tk], tfSpecFoiLim.time[tk])

                for chan in range(tfSpec.channel.size):

                    # Get reference channel in input object to determine underlying modulator
                    chanNo = chan
                    if select:
                        if "latency" not in select.keys():
                            chanNo = np.where(self.tfData.channel == select["channel"][chan])[0][0]
                    if chanNo % 2:
                        modIdx = self.odd[(-1)**trlNo]
                    else:
                        modIdx = self.even[(-1)**trlNo]
                    tfIdx[chanIdx] = chan
                    modulator = self.modulators[timeSelection, modIdx]
                    modCounts = [sum(modulator == modulator.min()), sum(modulator == modulator.max())]

                    # Be more lenient w/`tfSpec`: don't scan for min/max freq, but all peaks at once
                    # (auto-scale resolution potentially too coarse to differentiate b/w min/max);
                    # consider peak-count equal up to 2 misses
                    Zxx = trlArr[tuple(tfIdx)].squeeze()
                    ZxxMax = Zxx.max()
                    ZxxThresh = 0.2 * ZxxMax
                    _, freqPeaks = np.where(Zxx >= (ZxxMax - ZxxThresh))
                    peakVals, peakCounts = np.unique(freqPeaks, return_counts=True)
                    freqPeak = peakVals[peakCounts.argmax()]
                    modCount = np.ceil(sum(modCounts) / 2)
                    peakProfile = Zxx[:, freqPeak - 1 : freqPeak + 2].mean(axis=1)
                    peaks, _ = scisig.find_peaks(peakProfile, height=2*ZxxThresh, distance=5)
                    if np.abs(peaks.size - modCount) > 2:
                        modCount = sum(modCounts)
                    assert np.abs(peaks.size - modCount) <= 2

                    # Now for `tfSpecFoi`/`tfSpecFoiLim` on the other side be more
                    # stringent and really count maxima/minima (frequency values have
                    # been explicitly queried, must not be too coarse); that said,
                    # the peak-profile is quite rugged, so adjust `height` if necessary
                    for tfObj in [tfSpecFoi, tfSpecFoiLim]:
                        Zxx = tfObj.trials[tk][tuple(tfIdx)].squeeze()
                        ZxxMax = Zxx.max()
                        ZxxThresh = ZxxMax - 0.1 * ZxxMax
                        for fk, mFreq in enumerate(modFreqs):
                            freqPeak = np.where(np.abs(tfObj.freq - mFreq) < 1)[0][0]
                            peakProfile = Zxx[:, freqPeak - 1 : freqPeak + 2].mean(axis=1)
                            height = (1 - fk * 0.25) * ZxxThresh
                            peaks, _ = scisig.find_peaks(peakProfile, prominence=0.75*height, height=height, distance=5)
                            # if it doesn't fit, use a bigger hammer...
                            if np.abs(peaks.size - modCounts[fk]) > 2:
                                height = 0.9 * ZxxThresh
                                peaks, _ = scisig.find_peaks(peakProfile, prominence=0.75*height, height=height, distance=5)
                            assert np.abs(peaks.size - modCounts[fk]) <= 2

    def test_wav_toi(self):
        # Don't keep trials to speed things up a bit
        cfg = get_defaults(freqanalysis)
        cfg.method = "wavelet"
        cfg.wavelet = "Morlet"
        cfg.output = "pow"
        cfg.keeptrials = False

        # Test time-point arrays comprising onset, purely pre-onset, purely after
        # onset and non-unit spacing
        toiArrs = [np.arange(-2,7),
                   np.arange(-1, 6, 1/self.tfData.samplerate),
                   np.arange(1, 6, 2)]

        # Combine `toi`-testing w/in-place data-pre-selection
        for select in self.dataSelections:
            if select is not None and "latency" not in select.keys():
                cfg.select = select
                for toi in toiArrs:
                    cfg.toi = toi
                    tfSpec = freqanalysis(cfg, self.tfData)
                    assert np.allclose(cfg.toi, tfSpec.time[0])
                    assert tfSpec.samplerate == 1/(toi[1] - toi[0])

        # Test correct time-array assembly for ``toi = "all"`` (cut down data signifcantly
        # to not overflow memory here)
        cfg.select = {"trials": [0], "channel": [0], "latency": [-0.5, 0.5]}
        cfg.toi = "all"
        tfSpec = freqanalysis(cfg, TestWavelet.get_tfdata_wavelet())
        dt = 1/self.tfData.samplerate
        timeArr = np.arange(cfg.select["latency"][0], cfg.select["latency"][1] + dt, dt)
        assert np.allclose(tfSpec.time[0], timeArr)

        # Use `toi` array outside trial boundaries
        cfg.toi = self.tfData.time[0][:10]
        with pytest.raises(SPYValueError) as spyval:
            freqanalysis(cfg, TestWavelet.get_tfdata_wavelet())


        # Unsorted `toi` array
        cfg.toi = [0.3, -0.1, 0.2]
        with pytest.raises(SPYValueError):
            freqanalysis(cfg, TestSuperlet._get_tf_data_superlet())

    def test_wav_irregular_trials(self):
        # Set up wavelet to compute "full" TF spectrum for all time-points
        cfg = get_defaults(freqanalysis)
        cfg.method = "wavelet"
        cfg.wavelet = "Morlet"
        cfg.output = "pow"
        cfg.toi = "all"

        # start harmless: equidistant trials w/multiple tapers
        artdata = generate_artificial_data(nTrials=5, nChannels=16,
                                           equidistant=True, inmemory=False)
        tfSpec = freqanalysis(artdata, **cfg)
        for tk, origTime in enumerate(artdata.time):
            assert np.array_equal(origTime, tfSpec.time[tk])

        # non-equidistant trials w/multiple tapers
        artdata = generate_artificial_data(nTrials=5, nChannels=16,
                                           equidistant=False, inmemory=False)
        tfSpec = freqanalysis(artdata, **cfg)
        for tk, origTime in enumerate(artdata.time):
            assert np.array_equal(origTime, tfSpec.time[tk])

        # same + reversed dimensional order in input object
        cfg.data = generate_artificial_data(nTrials=5, nChannels=16,
                                            equidistant=False, inmemory=False,
                                            dimord=AnalogData._defaultDimord[::-1])
        tfSpec = freqanalysis(cfg)
        for tk, origTime in enumerate(cfg.data.time):
            assert np.array_equal(origTime, tfSpec.time[tk])

        # same + overlapping trials
        cfg.data = generate_artificial_data(nTrials=5, nChannels=16,
                                           equidistant=False, inmemory=False,
                                           dimord=AnalogData._defaultDimord[::-1],
                                           overlapping=True)
        tfSpec = freqanalysis(cfg)
        for tk, origTime in enumerate(cfg.data.time):
            assert np.array_equal(origTime, tfSpec.time[tk])

    @skip_low_mem
    def test_wav_parallel(self, testcluster):
        # collect all tests of current class and repeat them running concurrently
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and attr not in ["test_wav_parallel", "test_wav_cut_selections"])]
        for test in all_tests:
            getattr(self, test)()
            flush_local_cluster(testcluster)

        # now create uniform `cfg` for remaining SLURM tests
        cfg = StructDict()
        cfg.method = "wavelet"
        cfg.wavelet = "Morlet"
        cfg.output = "pow"
        cfg.toi = "all"

        nChannels = 3
        nTrials = 2

        # no. of HDF5 files that will make up virtual data-set in case of channel-chunking
        chanPerWrkr = 2
        nFiles = nTrials * (int(nChannels/chanPerWrkr) + int(nChannels % chanPerWrkr > 0))

        # simplest case: equidistant trial spacing, all in memory
        fileCount = [nTrials, nFiles]
        artdata = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                           inmemory=True)
        for k, chan_per_worker in enumerate([None, chanPerWrkr]):
            cfg.chan_per_worker = chan_per_worker
            tfSpec = freqanalysis(artdata, cfg)
            assert tfSpec.data.is_virtual
            assert len(tfSpec.data.virtual_sources()) == fileCount[k]

        # non-equidistant trial spacing
        cfg.keeptapers = False
        artdata = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                           inmemory=True, equidistant=False)
        for chan_per_worker in enumerate([None, chanPerWrkr]):
            tfSpec = freqanalysis(artdata, cfg)
            assert 1 > tfSpec.freq.min() > 0
            assert tfSpec.freq.max() == (self.tfData.samplerate / 2)

        # equidistant trial spacing
        cfg.output = "abs"
        artdata = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                           inmemory=False)
        for chan_per_worker in enumerate([None, chanPerWrkr]):
            tfSpec = freqanalysis(artdata, cfg)
            for tk, origTime in enumerate(artdata.time):
                assert np.array_equal(origTime, tfSpec.time[tk])

        # overlapping trial spacing, throw away trials
        cfg.keeptrials = "no"
        cfg.foilim = [1, 250]
        expectedFreqs = np.arange(1, cfg.foilim[1] + 1)
        artdata = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                           inmemory=False, equidistant=True,
                                           overlapping=True)
        tfSpec = freqanalysis(artdata, cfg)
        assert np.allclose(tfSpec.freq, expectedFreqs)
        assert tfSpec.data.shape == (tfSpec.time[0].size, 1, expectedFreqs.size, nChannels)

        client.close()

class TestSuperlet():

    @staticmethod
    def _get_tf_data_superlet():
        return _make_tf_signal(TestSuperlet.nChannels, TestSuperlet.nTrials, TestSuperlet.seed,
                                                           fadeIn=TestSuperlet.fadeIn, fadeOut=TestSuperlet.fadeOut)[0]


    # Prepare testing signal: ensure `fadeIn` and `fadeOut` are compatible w/`latency`
    # selection below
    nChannels = 4
    nTrials = 3
    seed = 151120
    fadeIn = -9.5
    fadeOut = 50.5
    tfData, modulators, even, odd, fader = _make_tf_signal(nChannels, nTrials, seed,
                                                           fadeIn=fadeIn, fadeOut=fadeOut)

    # Set up in-place data-selection dicts for the constructed object
    dataSelections = [None,
                      {"trials": [1, 2, 0],
                       "channel": ["channel" + str(i) for i in range(2, 4)][::-1]},
                      {"trials": [0, 2],
                       "channel": range(0, int(nChannels / 2)),
                       "latency": [-2, 6.8]}]

    # Helper function that reduces dataselections (keep `None` selection no matter what)
    def test_slet_cut_selections(self):
        self.dataSelections.pop(random.choice([-1, 1]))

    @skip_low_mem
    def test_slet_solution(self):

        # Compute TF specturm across entire time-interval (use integer-valued
        # time-points as wavelet centroids)
        cfg = get_defaults(freqanalysis)
        cfg.method = "superlet"
        cfg.order_max = 2
        cfg.output = "pow"

        # Set up index tuple for slicing computed TF spectra and collect values
        # of expected frequency peaks (for validation of `foi`/`foilim` selections below)
        chanIdx = SpectralData._defaultDimord.index("channel")
        tfIdx = [slice(None)] * len(SpectralData._defaultDimord)
        modFreqs = [330, 360]
        maxFreqs = np.hstack([np.arange(modFreqs[0] - 5, modFreqs[0] + 6),
                              np.arange(modFreqs[1] - 5, modFreqs[1] + 6)])
        foilimFreqs = np.arange(maxFreqs.min(), maxFreqs.max() + 1)

        for select in self.dataSelections:

            # Timing of `tfData` is identical for all trials, so to speed things up,
            # set up `timeArr` here - if `tfData` is modified, these computations have
            # to be moved inside the `enumerate(tfSpec.trials)`-loop!
            timeArr = np.arange(self.tfData.time[0][0], self.tfData.time[0][-1])
            if select:
                if "latency" in select.keys():
                    timeArr = np.arange(*select["latency"])
                    timeStart = int(select['latency'][0] * self.tfData.samplerate - self.tfData._t0[0])
                    timeStop = int(select['latency'][1] * self.tfData.samplerate - self.tfData._t0[0])
                    timeSelection = slice(timeStart, timeStop)
            else:
                timeSelection = np.where(self.fader == 1.0)[0]
            cfg.toi = timeArr

            # Skip below tests if `toi` and an in-place time-selection clash
            if select is not None and "latency" in select.keys():
                continue

            # Compute TF objects w\w/o`foi`/`foilim`
            cfg.select = select
            cfg.foi = maxFreqs
            tfSpecFoi = freqanalysis(cfg, self.tfData)
            cfg.foi = None
            assert np.allclose(tfSpecFoi.freq, maxFreqs)
            cfg.foilim = [maxFreqs.min(), maxFreqs.max()]
            tfSpecFoiLim = freqanalysis(cfg, self.tfData)
            cfg.foilim = None
            assert np.allclose(tfSpecFoiLim.freq, foilimFreqs)

            tfSpec = freqanalysis(cfg, self.tfData)
            assert 0.02 > tfSpec.freq.min() > 0
            assert tfSpec.freq.max() == (self.tfData.samplerate / 2)
            assert tfSpec.freq.size > 50

            for tk, _ in enumerate(tfSpecFoi.trials):

                # Get reference trial-number in input object
                trlNo = tk
                if select:
                    trlNo = select["trials"][tk]

                # Ensure timing array was computed correctly and independent of `foi`/`foilim`
                assert np.array_equal(timeArr, tfSpecFoi.time[tk])
                assert np.array_equal(tfSpecFoi.time[tk], tfSpecFoiLim.time[tk])
                assert np.array_equal(timeArr, tfSpec.time[tk])
                assert np.array_equal(tfSpec.time[tk], tfSpecFoi.time[tk])

                for chan in range(tfSpecFoi.channel.size):

                    # Get reference channel in input object to determine underlying modulator
                    chanNo = chan
                    if select:
                        if "latency" not in select.keys():
                            chanNo = np.where(self.tfData.channel == select["channel"][chan])[0][0]
                    if chanNo % 2:
                        modIdx = self.odd[(-1)**trlNo]
                    else:
                        modIdx = self.even[(-1)**trlNo]
                    tfIdx[chanIdx] = chan
                    modulator = self.modulators[timeSelection, modIdx]
                    modCounts = [sum(modulator == modulator.min()), sum(modulator == modulator.max())]

                    # Now for `tfSpecFoi`/`tfSpecFoiLim` on the other side be more
                    # stringent and really count maxima/minima (frequency values have
                    # been explicitly queried, must not be too coarse); that said,
                    # the peak-profile is quite rugged, so adjust `height` if necessary
                    for tfObj in [tfSpecFoi, tfSpecFoiLim]:
                        Zxx = tfObj.trials[tk][tuple(tfIdx)].squeeze()
                        ZxxMax = Zxx.max()
                        ZxxThresh = ZxxMax - 0.1 * ZxxMax
                        for fk, mFreq in enumerate(modFreqs):
                            freqPeak = np.where(np.abs(tfObj.freq - mFreq) < 1)[0][0]
                            peakProfile = Zxx[:, freqPeak - 1 : freqPeak + 2].mean(axis=1)
                            height = (1 - fk * 0.25) * ZxxThresh
                            peaks, _ = scisig.find_peaks(peakProfile, prominence=0.75*height, height=height, distance=5)
                            # if it doesn't fit, use a bigger hammer...
                            if np.abs(peaks.size - modCounts[fk]) > 2:
                                height = 0.9 * ZxxThresh
                                peaks, _ = scisig.find_peaks(peakProfile, prominence=0.75*height, height=height, distance=5)
                            assert np.abs(peaks.size - modCounts[fk]) <= 2

                    # Be more lenient w/`tfSpec`: don't scan for min/max freq, but all peaks at once
                    # (auto-scale resolution potentially too coarse to differentiate b/w min/max);

    def test_slet_toi(self):
        # Don't keep trials to speed things up a bit
        cfg = get_defaults(freqanalysis)
        cfg.method = "superlet"
        cfg.order_max = 2
        cfg.output = "pow"
        cfg.keeptrials = False

        # Test time-point arrays comprising onset, purely pre-onset, purely after
        # onset and non-unit spacing
        toiArrs = [np.arange(-2,7),
                   np.arange(-1, 6, 1/self.tfData.samplerate),
                   np.arange(1, 6, 2)]

        toiArrs = [random.choice(toiArrs)]

        # Combine `toi`-testing w/in-place data-pre-selection
        for select in self.dataSelections:
            if select is not None and "latency" not in select.keys():
                cfg.select = select
                for toi in toiArrs:
                    cfg.toi = toi
                    tfSpec = freqanalysis(cfg, self.tfData)
                    assert np.allclose(cfg.toi, tfSpec.time[0])
                    assert tfSpec.samplerate == 1/(toi[1] - toi[0])

        # Test correct time-array assembly for ``toi = "all"`` (cut down data signifcantly
        # to not overflow memory here)
        cfg.select = {"trials": [0], "channel": [0], "latency": [-0.5, 0.5]}
        cfg.toi = "all"
        tfSpec = freqanalysis(cfg, self.tfData)
        dt = 1/self.tfData.samplerate
        timeArr = np.arange(cfg.select["latency"][0], cfg.select["latency"][1] + dt, dt)
        assert np.allclose(tfSpec.time[0], timeArr)

        # Use `toi` array outside trial boundaries
        cfg.toi = self.tfData.time[0][:10]
        with pytest.raises(SPYValueError) as spyval:
            freqanalysis(cfg, TestSuperlet._get_tf_data_superlet())
            errmsg = "Invalid value of `toi`: expected all array elements to be bounded by {} and {}"
            assert errmsg.format(*cfg.select["latency"]) in str(spyval.value)

        # Unsorted `toi` array
        cfg.toi = [0.3, -0.1, 0.2]
        with pytest.raises(SPYValueError) as spyval:
            freqanalysis(cfg, TestSuperlet._get_tf_data_superlet())
            assert "Invalid value of `toi`: 'unsorted list/array'" in str(spyval.value)

    def test_slet_irregular_trials(self):
        # Set up wavelet to compute "full" TF spectrum for all time-points
        cfg = get_defaults(freqanalysis)
        cfg.method = "superlet"
        cfg.order_max = 2
        cfg.output = "pow"
        cfg.toi = "all"

        nTrials = 2
        nChannels = 2

        # start harmless: equidistant trials w/multiple tapers
        artdata = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                           equidistant=True, inmemory=False)
        tfSpec = freqanalysis(artdata, **cfg)
        for tk, origTime in enumerate(artdata.time):
            assert np.array_equal(origTime, tfSpec.time[tk])

        # non-equidistant trials w/multiple tapers
        artdata = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                           equidistant=False, inmemory=False)
        tfSpec = freqanalysis(artdata, **cfg)
        for tk, origTime in enumerate(artdata.time):
            assert np.array_equal(origTime, tfSpec.time[tk])

        # same + reversed dimensional order in input object
        cfg.data = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                            equidistant=False, inmemory=False,
                                            dimord=AnalogData._defaultDimord[::-1])
        tfSpec = freqanalysis(cfg)
        for tk, origTime in enumerate(cfg.data.time):
            assert np.array_equal(origTime, tfSpec.time[tk])

        # same + overlapping trials
        cfg.data = generate_artificial_data(nTrials=nTrials, nChannels=nChannels,
                                            equidistant=False, inmemory=False,
                                            dimord=AnalogData._defaultDimord[::-1],
                                            overlapping=True)
        tfSpec = freqanalysis(cfg)
        for tk, origTime in enumerate(cfg.data.time):
            assert np.array_equal(origTime, tfSpec.time[tk])

    @skip_low_mem
    def test_slet_parallel(self, testcluster):
        # collect all tests of current class and repeat them running concurrently
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and attr not in ["test_slet_parallel", "test_cut_slet_selections"])]
        for test in all_tests:
            getattr(self, test)()
            flush_local_cluster(testcluster)

        # now create uniform `cfg` for remaining SLURM tests
        cfg = StructDict()
        cfg.method = "superlet"
        cfg.order_max = 2
        cfg.output = "pow"
        cfg.toi = "all"

        # no. of HDF5 files that will make up virtual data-set in case of channel-chunking
        chanPerWrkr = 2
        nFiles = self.nTrials * (int(self.nChannels/chanPerWrkr)
                                 + int(self.nChannels % chanPerWrkr > 0))

        # simplest case: equidistant trial spacing, all in memory
        fileCount = [self.nTrials, nFiles]
        artdata = generate_artificial_data(nTrials=self.nTrials, nChannels=self.nChannels,
                                           inmemory=True)
        for k, chan_per_worker in enumerate([None, chanPerWrkr]):
            cfg.chan_per_worker = chan_per_worker
            tfSpec = freqanalysis(artdata, cfg)
            assert tfSpec.data.is_virtual
            assert len(tfSpec.data.virtual_sources()) == fileCount[k]

        # non-equidistant trial spacing
        cfg.keeptapers = False
        artdata = generate_artificial_data(nTrials=self.nTrials, nChannels=self.nChannels,
                                           inmemory=True, equidistant=False)
        for chan_per_worker in enumerate([None, chanPerWrkr]):
            tfSpec = freqanalysis(artdata, cfg)
            assert 1 > tfSpec.freq.min() > 0
            assert tfSpec.freq.max() == (self.tfData.samplerate / 2)

        # equidistant trial spacing
        cfg.output = "abs"
        artdata = generate_artificial_data(nTrials=self.nTrials, nChannels=self.nChannels,
                                           inmemory=False)
        for chan_per_worker in enumerate([None, chanPerWrkr]):
            tfSpec = freqanalysis(artdata, cfg)
            for tk, origTime in enumerate(artdata.time):
                assert np.array_equal(origTime, tfSpec.time[tk])

        # overlapping trial spacing, throw away trials
        cfg.keeptrials = "no"
        cfg.foilim = [1, 250]
        expectedFreqs = np.arange(1, cfg.foilim[1] + 1)
        artdata = generate_artificial_data(nTrials=self.nTrials, nChannels=self.nChannels,
                                           inmemory=False, equidistant=True,
                                           overlapping=True)
        tfSpec = freqanalysis(artdata, cfg)
        assert np.allclose(tfSpec.freq, expectedFreqs)
        assert tfSpec.data.shape == (tfSpec.time[0].size, 1, expectedFreqs.size, self.nChannels)

        client.close()


if __name__ == '__main__':
    T1 = TestMTMConvol()
    T2 = TestMTMFFT()
