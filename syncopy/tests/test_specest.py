# -*- coding: utf-8 -*-
#
# Test spectral estimation methods
#

# Builtin/3rd party package imports
import os
import tempfile
import inspect
import psutil
import gc
import pytest
import numpy as np
import scipy.signal as scisig
from numpy.lib.format import open_memmap
from syncopy import __acme__
if __acme__:
    import dask.distributed as dd

# Local imports
from syncopy.tests.misc import generate_artificial_data, flush_local_cluster
from syncopy.specest.freqanalysis import freqanalysis
from syncopy.shared.errors import SPYValueError
from syncopy.datatype.base_data import VirtualData, Selector
from syncopy.datatype import AnalogData, SpectralData
from syncopy.shared.tools import StructDict, get_defaults

# Decorator to decide whether or not to run dask-related tests
skip_without_acme = pytest.mark.skipif(not __acme__, reason="acme not available")

# Decorator to decide whether or not to run memory-intensive tests
availMem = psutil.virtual_memory().total
skip_low_mem = pytest.mark.skipif(availMem < 10 * 1024**3, reason="less than 10GB RAM available")


# Local helper for constructing TF testing signals
def _make_tf_signal(nChannels, nTrials, seed, fadeIn=None, fadeOut=None):

    # Construct high-frequency signal modulated by slow oscillating cosine and
    # add time-decaying noise
    nChan2 = int(nChannels / 2)
    fs = 1000
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    numType = "float32"
    modPeriods = [0.125, 0.0625]
    rng = np.random.default_rng(seed)
    tStart = -29.5
    tStop = 70.5
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
    fadeOut = np.arange((fadeOut - tStart) * fs, 100 * fs, dtype=np.intp)
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
             416.,  32., 438., 111., 140., 304., 327., 494.,  23., 493.]
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
                          "channels": ["channel" + str(i) for i in range(12, 28)][::-1]},
                         {"trials": [0, 1, 2],
                          "channels": range(0, int(nChannels / 2)),
                          "toilim": [0.25, 0.75]}]

    # Data selections to be tested w/`artdata` generated below (use fixed but arbitrary
    # random number seed to randomly select time-points for `toi` (with repetitions)
    seed = np.random.RandomState(13)
    artdataSelections = [None,
                         {"trials": [3, 1, 0],
                          "channels": ["channel" + str(i) for i in range(10, 15)][::-1],
                          "toi": None},
                         {"trials": [0, 1, 2],
                          "channels": range(0, 8),
                          "toilim": [-0.5, 0.6]}]

    # Error tolerances for target amplitudes (depend on data selection!)
    tols = [1, 1, 1.5]

    # Error tolerance for frequency-matching
    ftol = 0.25

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


    def test_allocout(self):
        # call `freqanalysis` w/pre-allocated output object
        out = SpectralData(dimord=SpectralData._defaultDimord)
        freqanalysis(self.adata, method="mtmfft", taper="hann", out=out)
        assert len(out.trials) == self.nTrials
        assert out.taper.size == 1
        assert out.freq.size == self.fband.size + 1
        assert np.allclose([0] + self.fband.tolist(), out.freq)
        assert out.channel.size == self.nChannels

        # build `cfg` object for calling
        cfg = StructDict()
        cfg.method = "mtmfft"
        cfg.taper = "hann"
        cfg.keeptrials = "no"
        cfg.output = "abs"
        cfg.out = SpectralData(dimord=SpectralData._defaultDimord)

        # throw away trials
        freqanalysis(self.adata, cfg)
        assert len(cfg.out.time) == 1
        assert len(cfg.out.time[0]) == 1
        assert np.all(cfg.out.sampleinfo == [0, 1])
        assert cfg.out.data.shape[0] == 1  # ensure trial-count == 1

        # keep trials but throw away tapers
        out = SpectralData(dimord=SpectralData._defaultDimord)
        freqanalysis(self.adata, method="mtmfft", taper="dpss",
                     tapsmofrq=3, keeptapers=False, output="pow", out=out)
        assert out.sampleinfo.shape == (self.nTrials, 2)
        assert out.taper.size == 1

        # re-use `cfg` from above and additionally throw away `tapers`
        cfg.dataset = self.adata
        cfg.out = SpectralData(dimord=SpectralData._defaultDimord)
        cfg.taper = "dpss"
        cfg.tapsmofrq = 3
        cfg.output = "pow"
        cfg.keeptapers = False
        freqanalysis(cfg)
        assert cfg.out.taper.size == 1

    def test_solution(self):
        # ensure channel-specific frequencies are identified correctly
        for sk, select in enumerate(self.sigdataSelections):
            sel = Selector(self.adata, select)
            spec = freqanalysis(self.adata, method="mtmfft", taper="hann",
                                pad_to_length="nextpow2", output="pow", select=select)

            chanList = np.arange(self.nChannels)[sel.channel]
            amps = np.empty((len(sel.trials) * len(chanList),))
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

    def test_foi(self):
        for select in self.sigdataSelections:

            # `foi` lims outside valid bounds
            with pytest.raises(SPYValueError):
                freqanalysis(self.adata, method="mtmfft", taper="hann",
                             foi=[-0.5, self.fs / 3], select=select)
            with pytest.raises(SPYValueError):
                freqanalysis(self.adata, method="mtmfft", taper="hann",
                             foi=[1, self.fs], select=select)

            foi = self.fband[1:int(self.fband.size / 3)]

            # offset `foi` by 0.1 Hz - resulting freqs must be unaffected
            ftmp = foi + 0.1
            spec = freqanalysis(self.adata, method="mtmfft", taper="hann",
                                pad_to_length="nextpow2", foi=ftmp, select=select)
            assert np.all(spec.freq == foi)

            # unsorted, duplicate entries in `foi` - result must stay the same
            ftmp = np.hstack([foi, np.full(20, foi[0])])
            spec = freqanalysis(self.adata, method="mtmfft", taper="hann",
                                pad_to_length="nextpow2", foi=ftmp, select=select)
            assert np.all(spec.freq == foi)

    def test_dpss(self):

        for select in self.sigdataSelections:
            sel = Selector(self.adata, select)
            chanList = np.arange(self.nChannels)[sel.channel]

            # ensure default setting results in single taper
            spec = freqanalysis(self.adata, method="mtmfft",
                                taper="dpss", tapsmofrq=3,  output="pow", select=select)
            assert spec.taper.size == 1
            assert spec.channel.size == len(chanList)

            # specify tapers
            spec = freqanalysis(self.adata, method="mtmfft", taper="dpss",
                                tapsmofrq=7, keeptapers=True, select=select)
            assert spec.channel.size == len(chanList)

        # non-equidistant data w/multiple tapers
        artdata = generate_artificial_data(nTrials=5, nChannels=16,
                                           equidistant=False, inmemory=False)
        timeAxis = artdata.dimord.index("time")
        cfg = StructDict()
        cfg.method = "mtmfft"
        cfg.taper = "dpss"
        cfg.tapsmofrq = 9.3
        cfg.output = "pow"

        for select in self.artdataSelections:

            # unsorted, w/repetitions, do not pad
            if select is not None and "toi" in select.keys():
                select["toi"] = self.seed.choice(artdata.time[0], int(artdata.time[0].size))
            sel = Selector(artdata, select)
            cfg.select = select
            spec = freqanalysis(cfg, artdata)

            # ensure correctness of padding (respecting min. trial length + time-selection)
            if select is None:
                maxtrlno = np.diff(artdata.sampleinfo).argmax()
                nSamples = artdata.trials[maxtrlno].shape[timeAxis]
            elif "toi" in select:
                nSamples = len(select["toi"])
            else:
                nSamples = artdata.time[sel.trials[0]][sel.time[0]].size
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

            # unsorted, w/repetitions, do not pad
            if select is not None and "toi" in select.keys():
                select["toi"] = self.seed.choice(cfg.data.time[0], int(cfg.data.time[0].size))
            sel = Selector(cfg.data, select)
            cfg.select = select

            spec = freqanalysis(cfg)

            # ensure correctness of padding (respecting min. trial length + time-selection)
            if select is None:
                maxtrlno = np.diff(cfg.data.sampleinfo).argmax()
                nSamples = cfg.data.trials[maxtrlno].shape[timeAxis]
            elif "toi" in select:
                nSamples = len(select["toi"])
            else:
                nSamples = cfg.data.time[sel.trials[0]][sel.time[0]].size
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

            # unsorted, w/repetitions, do not pad
            # cfg.pop("pad", None)
            if select is not None and "toi" in select.keys():
                select["toi"] = self.seed.choice(cfg.data.time[0], int(cfg.data.time[0].size))
            sel = Selector(cfg.data, select)
            cfg.select = select

            spec = freqanalysis(cfg)

            # ensure correctness of padding (respecting min. trial length + time-selection)
            if select is None:
                maxtrlno = np.diff(cfg.data.sampleinfo).argmax()
                nSamples = cfg.data.trials[maxtrlno].shape[timeAxis]
            elif "toi" in select:
                nSamples = len(select["toi"])
            else:
                nSamples = cfg.data.time[sel.trials[0]][sel.time[0]].size
            freqs = np.fft.rfftfreq(nSamples, 1 / cfg.data.samplerate)
            assert spec.freq.size == freqs.size
            assert np.max(spec.freq - freqs) < self.ftol
            assert spec.taper.size == 1


    @pytest.mark.skip(reason="VirtualData is currently not supported")
    def test_vdata(self):
        # test constant padding w/`VirtualData` objects (trials have identical lengths)
        with tempfile.TemporaryDirectory() as tdir:
            npad = 10
            fname = os.path.join(tdir, "dummy.npy")
            np.save(fname, self.sig)
            dmap = open_memmap(fname, mode="r")
            vdata = VirtualData([dmap, dmap])
            avdata = AnalogData(vdata, samplerate=self.fs,
                                trialdefinition=self.trialdefinition)
            spec = freqanalysis(avdata, method="mtmfft", taper="dpss",
                                tapsmofrq=3, keeptapers=False, output="abs", pad="relative",
                                padlength=npad)
            assert (np.diff(avdata.sampleinfo)[0][0] + npad) / 2 + 1 == spec.freq.size
            del avdata, vdata, dmap, spec
            gc.collect()  # force-garbage-collect object so that tempdir can be closed

    @skip_without_acme
    @skip_low_mem
    def test_parallel(self, testcluster):
        # collect all tests of current class and repeat them using dask
        # (skip VirtualData tests since ``wrapper_io`` expects valid headers)
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and attr != "test_parallel")]
        all_tests.remove("test_vdata")
        for test in all_tests:
            getattr(self, test)()
            flush_local_cluster(testcluster)

        # now create uniform `cfg` for remaining SLURM tests
        cfg = StructDict()
        cfg.method = "mtmfft"
        cfg.taper = "dpss"
        cfg.tapsmofrq = 9.3
        cfg.output = "pow"

        # no. of HDF5 files that will make up virtual data-set in case of channel-chunking
        chanPerWrkr = 7
        nFiles = self.nTrials * (int(self.nChannels/chanPerWrkr)
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

    # Data selection dict for the above object
    dataSelections = [None,
                      {"trials": [1, 2, 0],
                       "channels": ["channel" + str(i) for i in range(2, 6)][::-1]},
                      {"trials": [0, 2],
                       "channels": range(0, nChan2),
                       "toilim": [-20, 60.8]}]

    def test_tf_output(self):
        # Set up basic TF analysis parameters to not slow down things too much
        cfg = get_defaults(freqanalysis)
        cfg.method = "mtmconvol"
        cfg.taper = "hann"
        cfg.toi = np.linspace(-20, 60, 10)
        cfg.t_ftimwin = 1.0

        for select in self.dataSelections:
            select = self.dataSelections[-1]
            cfg.select = select
            cfg.output = "fourier"
            tfSpec = freqanalysis(cfg, self.tfData)
            assert "complex" in tfSpec.data.dtype.name
            cfg.output = "abs"
            tfSpec = freqanalysis(cfg, self.tfData)
            assert "float" in tfSpec.data.dtype.name
            cfg.output = "pow"
            tfSpec = freqanalysis(cfg, self.tfData)
            assert "float" in tfSpec.data.dtype.name

    def test_tf_allocout(self):
        # use `mtmconvol` w/pre-allocated output object
        out = SpectralData(dimord=SpectralData._defaultDimord)
        freqanalysis(self.tfData, method="mtmconvol", taper="hann", toi=0.0,
                     t_ftimwin=1.0, out=out)
        assert len(out.trials) == len(self.tfData.trials)
        assert out.taper.size == 1
        assert out.freq.size == self.tfData.samplerate / 2 + 1
        assert out.channel.size == self.nChannels

        # build `cfg` object for calling
        cfg = StructDict()
        cfg.method = "mtmconvol"
        cfg.taper = "hann"
        cfg.keeptrials = "no"
        cfg.output = "abs"
        cfg.toi = 0.0
        cfg.t_ftimwin = 1.0
        cfg.out = SpectralData(dimord=SpectralData._defaultDimord)

        # throw away trials: computing `trLen` this way only works for non-overlapping windows!
        freqanalysis(self.tfData, cfg)
        assert len(cfg.out.time) == 1
        trLen = len(self.tfData.time[0]) / (cfg.t_ftimwin * self.tfData.samplerate)
        assert len(cfg.out.time[0]) == trLen
        assert np.all(cfg.out.sampleinfo == [0, trLen])
        assert cfg.out.data.shape[0] == trLen  # ensure trial-count == 1

        # keep trials but throw away tapers
        out = SpectralData(dimord=SpectralData._defaultDimord)
        freqanalysis(self.tfData, method="mtmconvol", taper="dpss", tapsmofrq=3,
                     keeptapers=False, output="pow", toi=0.0, t_ftimwin=1.0,
                     out=out)
        assert out.sampleinfo.shape == (self.nTrials, 2)
        assert out.taper.size == 1

        # re-use `cfg` from above and additionally throw away `tapers`
        cfg.dataset = self.tfData
        cfg.out = SpectralData(dimord=SpectralData._defaultDimord)
        cfg.taper = "dpss"
        cfg.tapsmofrq = 3
        cfg.keeptapers = False
        cfg.output = "pow"
        freqanalysis(cfg)
        assert cfg.out.taper.size == 1

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
            tfSpec = freqanalysis(cfg, self.tfData)
            cfg.foi = maxFreqs
            tfSpecFoi = freqanalysis(cfg, self.tfData)
            cfg.foi = None
            cfg.foilim = [maxFreqs.min(), maxFreqs.max()]
            tfSpecFoiLim = freqanalysis(cfg, self.tfData)
            cfg.foilim = None

            # Ensure TF objects contain expected/requested frequencies
            assert np.array_equal(tfSpec.freq, allFreqs)
            assert np.array_equal(tfSpecFoi.freq, maxFreqs)
            assert np.array_equal(tfSpecFoiLim.freq, foilimFreqs)

            for tk, trlArr in enumerate(tfSpec.trials):

                # Compute expected timing array depending on `toilim`
                trlNo = tk
                timeArr = np.arange(self.tfData.time[trlNo][0], self.tfData.time[trlNo][-1])
                timeSelection = slice(None)
                if select:
                    trlNo = select["trials"][tk]
                    if "toilim" in select.keys():
                        timeArr = np.arange(*select["toilim"])
                        timeStart = int(select['toilim'][0] * self.tfData.samplerate - self.tfData._t0[trlNo])
                        timeStop = int(select['toilim'][1] * self.tfData.samplerate - self.tfData._t0[trlNo])
                        timeSelection = slice(timeStart, timeStop)

                # Ensure timing array was computed correctly and independent of `foi`/`foilim`
                assert np.array_equal(timeArr, tfSpec.time[tk])
                assert np.array_equal(tfSpec.time[tk], tfSpecFoi.time[tk])
                assert np.array_equal(tfSpecFoi.time[tk], tfSpecFoiLim.time[tk])

                for chan in range(tfSpec.channel.size):

                    # Get reference channel in input object to determine underlying modulator
                    chanNo = chan
                    if select:
                        if "toilim" not in select.keys():
                            chanNo = np.where(self.tfData.channel == select["channels"][chan])[0][0]
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
        toiArrs = [np.arange(-10, 15.1),
                   np.arange(-15, -10, 1/self.tfData.samplerate),
                   np.arange(1, 20, 2)]
        winSizes = [0.5, 1.0]

        # Combine `toi`-testing w/in-place data-pre-selection
        for select in self.dataSelections:
            cfg.select = select
            tStart = self.tfData.time[0][0]
            tStop = self.tfData.time[0][-1]
            if select:
                if "toilim" in select.keys():
                    tStart = select["toilim"][0]
                    tStop = select["toilim"][1]

            # Test TF calculation w/different window-size/-centroids: ensure
            # resulting timing arrays are correct
            for winsize in winSizes:
                cfg.t_ftimwin = winsize
                for toi in toiVals:
                    cfg.toi = toi
                    tfSpec = freqanalysis(cfg, self.tfData)
                    tStep = winsize - toi * winsize
                    timeArr = np.arange(tStart, tStop, tStep)
                    assert np.allclose(timeArr, tfSpec.time[0])

            # Test window-centroids specified as time-point arrays
            cfg.t_ftimwin = 0.05
            for toi in toiArrs:
                cfg.toi = toi
                tfSpec = freqanalysis(cfg, self.tfData)
                assert np.allclose(cfg.toi, tfSpec.time[0])
                assert tfSpec.samplerate == 1/(toi[1] - toi[0])

            # Unevenly sampled array: timing currently in lala-land, but sizes must match
            cfg.toi = [-5, 3, 10]
            tfSpec = freqanalysis(cfg, self.tfData)
            assert tfSpec.time[0].size == len(cfg.toi)

        # Test correct time-array assembly for ``toi = "all"`` (cut down data signifcantly
        # to not overflow memory here); same for ``toi = 1.0```
        cfg.taper = "dpss"
        cfg.tapsmofrq = 10
        cfg.keeptapers = True
        cfg.select = {"trials": [0], "channels": [0], "toilim": [-0.5, 0.5]}
        cfg.toi = "all"
        cfg.t_ftimwin = 0.05
        tfSpec = freqanalysis(cfg, self.tfData)
        assert tfSpec.taper.size >= 1
        dt = 1 / self.tfData.samplerate
        timeArr = np.arange(cfg.select["toilim"][0], cfg.select["toilim"][1] + dt, dt)
        assert np.allclose(tfSpec.time[0], timeArr)
        cfg.toi = 1.0
        tfSpec = freqanalysis(cfg, self.tfData)
        assert np.allclose(tfSpec.time[0], timeArr)

        # Use a window-size larger than the pre-selected interval defined above
        cfg.t_ftimwin = 5.0
        with pytest.raises(SPYValueError) as spyval:
            freqanalysis(cfg, self.tfData)
            assert "Invalid value of `t_ftimwin`" in str(spyval.value)
        cfg.t_ftimwin = 0.05

        # Use `toi` array outside trial boundaries
        cfg.toi = self.tfData.time[0][:10]
        with pytest.raises(SPYValueError) as spyval:
            freqanalysis(cfg, self.tfData)
            errmsg = "Invalid value of `toi`: expected all array elements to be bounded by {} and {}"
            assert errmsg.format(*cfg.select["toilim"]) in str(spyval.value)

        # Unsorted `toi` array
        cfg.toi = [0.3, -0.1, 0.2]
        with pytest.raises(SPYValueError) as spyval:
            freqanalysis(cfg, self.tfData)
            assert "Invalid value of `toi`: 'unsorted list/array'" in str(spyval.value)

    def test_tf_irregular_trials(self):
        # Settings for computing "full" non-overlapping TF-spectrum with DPSS tapers:
        # ensure non-equidistant/overlapping trials are processed (padded) correctly
        # also make sure ``toi = "all"`` works under any circumstance
        cfg = get_defaults(freqanalysis)
        cfg.method = "mtmconvol"
        cfg.taper = "dpss"
        cfg.tapsmofrq = 10
        cfg.t_ftimwin = 1.0
        cfg.output = "pow"
        cfg.keeptapers = True

        # start harmless: equidistant trials w/multiple tapers
        cfg.toi = 0.0
        artdata = generate_artificial_data(nTrials=5, nChannels=16,
                                           equidistant=True, inmemory=False)
        tfSpec = freqanalysis(artdata, **cfg)
        assert tfSpec.taper.size >= 1
        for tk, origTime in enumerate(artdata.time):
            assert np.array_equal(np.unique(np.floor(origTime)), tfSpec.time[tk])

        # to process all time-points via `stft`, reduce dataset size (avoid oom kills)
        cfg.toi = "all"
        artdata = generate_artificial_data(nTrials=5, nChannels=4,
                                           equidistant=True, inmemory=False)
        tfSpec = freqanalysis(artdata, **cfg)
        for tk, origTime in enumerate(artdata.time):
            assert np.array_equal(origTime, tfSpec.time[tk])

        # non-equidistant trials w/multiple tapers
        cfg.toi = 0.0
        artdata = generate_artificial_data(nTrials=5, nChannels=8,
                                           equidistant=False, inmemory=False)
        tfSpec = freqanalysis(artdata, **cfg)
        assert tfSpec.taper.size >= 1
        for tk, origTime in enumerate(artdata.time):
            assert np.array_equal(np.unique(np.floor(origTime)), tfSpec.time[tk])
        cfg.toi = "all"
        tfSpec = freqanalysis(artdata, **cfg)
        for tk, origTime in enumerate(artdata.time):
            assert np.array_equal(origTime, tfSpec.time[tk])

        # same + reversed dimensional order in input object
        cfg.toi = 0.0
        cfg.data = generate_artificial_data(nTrials=5, nChannels=8,
                                            equidistant=False, inmemory=False,
                                            dimord=AnalogData._defaultDimord[::-1])
        tfSpec = freqanalysis(cfg)
        assert tfSpec.taper.size >= 1
        for tk, origTime in enumerate(cfg.data.time):
            assert np.array_equal(np.unique(np.floor(origTime)), tfSpec.time[tk])
        cfg.toi = "all"
        tfSpec = freqanalysis(cfg)
        for tk, origTime in enumerate(cfg.data.time):
            assert np.array_equal(origTime, tfSpec.time[tk])

        # same + overlapping trials
        cfg.toi = 0.0
        cfg.data = generate_artificial_data(nTrials=5, nChannels=4,
                                            equidistant=False, inmemory=False,
                                            dimord=AnalogData._defaultDimord[::-1],
                                            overlapping=True)
        tfSpec = freqanalysis(cfg)
        assert tfSpec.taper.size >= 1
        for tk, origTime in enumerate(cfg.data.time):
            assert np.array_equal(np.unique(np.floor(origTime)), tfSpec.time[tk])
        cfg.toi = "all"
        tfSpec = freqanalysis(cfg)
        for tk, origTime in enumerate(cfg.data.time):
            assert np.array_equal(origTime, tfSpec.time[tk])

    @skip_without_acme
    @skip_low_mem
    def test_tf_parallel(self, testcluster):
        # collect all tests of current class and repeat them running concurrently
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and attr != "test_tf_parallel")]
        for test in all_tests:
            getattr(self, test)()
            flush_local_cluster(testcluster)

        # now create uniform `cfg` for remaining SLURM tests
        cfg = StructDict()
        cfg.method = "mtmconvol"
        cfg.taper = "hann"
        cfg.t_ftimwin = 1.0
        cfg.toi = 0
        cfg.output = "pow"

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
        expectedFreqs = np.arange(artdata.samplerate / 2 + 1)
        for chan_per_worker in enumerate([None, chanPerWrkr]):
            tfSpec = freqanalysis(artdata, cfg)
            assert np.array_equal(tfSpec.freq, expectedFreqs)
            assert tfSpec.taper.size == 1

        # equidistant trial spacing, keep tapers
        cfg.output = "abs"
        cfg.taper = "dpss"
        cfg.tapsmofrq = 10
        cfg.keeptapers = True
        artdata = generate_artificial_data(nTrials=self.nTrials, nChannels=self.nChannels,
                                           inmemory=False)
        for chan_per_worker in enumerate([None, chanPerWrkr]):
            tfSpec = freqanalysis(artdata, cfg)
            assert tfSpec.taper.size >= 1

        # overlapping trial spacing, throw away trials and tapers
        cfg.keeptapers = False
        cfg.keeptrials = "no"
        cfg.output = "pow"
        artdata = generate_artificial_data(nTrials=self.nTrials, nChannels=self.nChannels,
                                           inmemory=False, equidistant=True,
                                           overlapping=True)
        expectedFreqs = np.arange(artdata.samplerate / 2 + 1)
        tfSpec = freqanalysis(artdata, cfg)
        assert np.array_equal(tfSpec.freq, expectedFreqs)
        assert tfSpec.taper.size == 1
        assert np.array_equal(np.unique(np.floor(artdata.time[0])), tfSpec.time[0])
        assert tfSpec.data.shape == (tfSpec.time[0].size, 1, expectedFreqs.size, self.nChannels)

        client.close()


class TestWavelet():

    # Prepare testing signal: ensure `fadeIn` and `fadeOut` are compatible w/`toilim`
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
                       "channels": ["channel" + str(i) for i in range(2, 4)][::-1]},
                      {"trials": [0, 2],
                       "channels": range(0, int(nChannels / 2)),
                       "toilim": [-20, 60.8]}]

    @skip_low_mem
    def test_wav_solution(self):

        # Compute TF specturm across entire time-interval (use integer-valued
        # time-points as wavelet centroids)
        cfg = get_defaults(freqanalysis)
        cfg.method = "wavelet"
        cfg.wavelet = "Morlet"
        cfg.width = 1
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
                if "toilim" in select.keys():
                    timeArr = np.arange(*select["toilim"])
                    timeStart = int(select['toilim'][0] * self.tfData.samplerate - self.tfData._t0[0])
                    timeStop = int(select['toilim'][1] * self.tfData.samplerate - self.tfData._t0[0])
                    timeSelection = slice(timeStart, timeStop)
            else:
                timeSelection = np.where(self.fader == 1.0)[0]
            cfg.toi = timeArr

            # Compute TF objects w\w/o`foi`/`foilim`
            cfg.select = select
            tfSpec = freqanalysis(cfg, self.tfData)
            cfg.foi = maxFreqs
            tfSpecFoi = freqanalysis(cfg, self.tfData)
            cfg.foi = None
            cfg.foilim = [maxFreqs.min(), maxFreqs.max()]
            tfSpecFoiLim = freqanalysis(cfg, self.tfData)
            cfg.foilim = None

            # Ensure TF objects contain expected/requested frequencies
            assert 0.02 > tfSpec.freq.min() > 0
            assert tfSpec.freq.max() == (self.tfData.samplerate / 2)
            assert tfSpec.freq.size > 60
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
                        if "toilim" not in select.keys():
                            chanNo = np.where(self.tfData.channel == select["channels"][chan])[0][0]
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
        toiArrs = [np.arange(-10, 15.1),
                   np.arange(-15, -10, 1/self.tfData.samplerate),
                   np.arange(1, 20, 2)]

        # Combine `toi`-testing w/in-place data-pre-selection
        for select in self.dataSelections:
            cfg.select = select
            for toi in toiArrs:
                cfg.toi = toi
                tfSpec = freqanalysis(cfg, self.tfData)
                assert np.allclose(cfg.toi, tfSpec.time[0])
                assert tfSpec.samplerate == 1/(toi[1] - toi[0])

        # Test correct time-array assembly for ``toi = "all"`` (cut down data signifcantly
        # to not overflow memory here)
        cfg.select = {"trials": [0], "channels": [0], "toilim": [-0.5, 0.5]}
        cfg.toi = "all"
        tfSpec = freqanalysis(cfg, self.tfData)
        dt = 1/self.tfData.samplerate
        timeArr = np.arange(cfg.select["toilim"][0], cfg.select["toilim"][1] + dt, dt)
        assert np.allclose(tfSpec.time[0], timeArr)

        # Use `toi` array outside trial boundaries
        cfg.toi = self.tfData.time[0][:10]
        with pytest.raises(SPYValueError) as spyval:
            freqanalysis(cfg, self.tfData)
            errmsg = "Invalid value of `toi`: expected all array elements to be bounded by {} and {}"
            assert errmsg.format(*cfg.select["toilim"]) in str(spyval.value)

        # Unsorted `toi` array
        cfg.toi = [0.3, -0.1, 0.2]
        with pytest.raises(SPYValueError) as spyval:
            freqanalysis(cfg, self.tfData)
            assert "Invalid value of `toi`: 'unsorted list/array'" in str(spyval.value)

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

    @skip_without_acme
    @skip_low_mem
    def test_wav_parallel(self, testcluster):
        # collect all tests of current class and repeat them running concurrently
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and attr != "test_wav_parallel")]
        for test in all_tests:
            getattr(self, test)()
            flush_local_cluster(testcluster)

        # now create uniform `cfg` for remaining SLURM tests
        cfg = StructDict()
        cfg.method = "wavelet"
        cfg.wavelet = "Morlet"
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


class TestSuperlet():

    # Prepare testing signal: ensure `fadeIn` and `fadeOut` are compatible w/`toilim`
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
                       "channels": ["channel" + str(i) for i in range(2, 4)][::-1]},
                      {"trials": [0, 2],
                       "channels": range(0, int(nChannels / 2)),
                       "toilim": [-20, 60.8]}]

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
                if "toilim" in select.keys():
                    timeArr = np.arange(*select["toilim"])
                    timeStart = int(select['toilim'][0] * self.tfData.samplerate - self.tfData._t0[0])
                    timeStop = int(select['toilim'][1] * self.tfData.samplerate - self.tfData._t0[0])
                    timeSelection = slice(timeStart, timeStop)
            else:
                timeSelection = np.where(self.fader == 1.0)[0]
            cfg.toi = timeArr

            # Compute TF objects w\w/o`foi`/`foilim`
            cfg.select = select
            tfSpec = freqanalysis(cfg, self.tfData)
            cfg.foi = maxFreqs
            tfSpecFoi = freqanalysis(cfg, self.tfData)
            cfg.foi = None
            cfg.foilim = [maxFreqs.min(), maxFreqs.max()]
            tfSpecFoiLim = freqanalysis(cfg, self.tfData)
            cfg.foilim = None

            # Ensure TF objects contain expected/requested frequencies
            assert 0.02 > tfSpec.freq.min() > 0
            assert tfSpec.freq.max() == (self.tfData.samplerate / 2)
            assert tfSpec.freq.size > 60
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
                        if "toilim" not in select.keys():
                            chanNo = np.where(self.tfData.channel == select["channels"][chan])[0][0]
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

    def test_slet_toi(self):
        # Don't keep trials to speed things up a bit
        cfg = get_defaults(freqanalysis)
        cfg.method = "superlet"
        cfg.order_max = 2
        cfg.output = "pow"
        cfg.keeptrials = False

        # Test time-point arrays comprising onset, purely pre-onset, purely after
        # onset and non-unit spacing
        toiArrs = [np.arange(-10, 15.1),
                   np.arange(-15, -10, 1/self.tfData.samplerate),
                   np.arange(1, 20, 2)]

        # Combine `toi`-testing w/in-place data-pre-selection
        for select in self.dataSelections:
            cfg.select = select
            for toi in toiArrs:
                cfg.toi = toi
                tfSpec = freqanalysis(cfg, self.tfData)
                assert np.allclose(cfg.toi, tfSpec.time[0])
                assert tfSpec.samplerate == 1/(toi[1] - toi[0])

        # Test correct time-array assembly for ``toi = "all"`` (cut down data signifcantly
        # to not overflow memory here)
        cfg.select = {"trials": [0], "channels": [0], "toilim": [-0.5, 0.5]}
        cfg.toi = "all"
        tfSpec = freqanalysis(cfg, self.tfData)
        dt = 1/self.tfData.samplerate
        timeArr = np.arange(cfg.select["toilim"][0], cfg.select["toilim"][1] + dt, dt)
        assert np.allclose(tfSpec.time[0], timeArr)

        # Use `toi` array outside trial boundaries
        cfg.toi = self.tfData.time[0][:10]
        with pytest.raises(SPYValueError) as spyval:
            freqanalysis(cfg, self.tfData)
            errmsg = "Invalid value of `toi`: expected all array elements to be bounded by {} and {}"
            assert errmsg.format(*cfg.select["toilim"]) in str(spyval.value)

        # Unsorted `toi` array
        cfg.toi = [0.3, -0.1, 0.2]
        with pytest.raises(SPYValueError) as spyval:
            freqanalysis(cfg, self.tfData)
            assert "Invalid value of `toi`: 'unsorted list/array'" in str(spyval.value)

    def test_slet_irregular_trials(self):
        # Set up wavelet to compute "full" TF spectrum for all time-points
        cfg = get_defaults(freqanalysis)
        cfg.method = "superlet"
        cfg.order_max = 2
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

    @skip_without_acme
    @skip_low_mem
    def test_slet_parallel(self, testcluster):
        # collect all tests of current class and repeat them running concurrently
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and attr != "test_slet_parallel")]
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
