# -*- coding: utf-8 -*-
#
# Test spectral estimation methods
#
# Created: 2019-06-17 09:45:47
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-15 12:05:40>

from syncopy.tests.misc import generate_artifical_data
from syncopy.specest.freqanalysis import freqanalysis
from syncopy.shared.errors import SPYValueError
from syncopy.datatype.data_methods import _nextpow2
from syncopy.datatype.base_data import VirtualData, Selector
from syncopy.datatype import AnalogData, SpectralData, StructDict, padding
import os
import tempfile
import inspect
import gc
import pytest
import numpy as np
from numpy.lib.format import open_memmap
from syncopy import __dask__
if __dask__:
    import dask.distributed as dd

# Decorator to decide whether or not to run dask-related tests
skip_without_dask = pytest.mark.skipif(
    not __dask__, reason="dask not available")


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
                          "toilim": [1., 1.5]}]

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

    def test_solution(self):
        # ensure channel-specific frequencies are identified correctly
        for sk, select in enumerate(self.sigdataSelections):
            sel = Selector(self.adata, select)
            spec = freqanalysis(self.adata, method="mtmfft", taper="hann",
                                output="pow", select=select)

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
                             foi=[0.5, self.fs / 3], select=select)
            with pytest.raises(SPYValueError):
                freqanalysis(self.adata, method="mtmfft", taper="hann",
                             foi=[1, self.fs], select=select)

            foi = self.fband[1:int(self.fband.size / 3)]

            # offset `foi` by 0.1 Hz - resulting freqs must be unaffected
            ftmp = foi + 0.1
            spec = freqanalysis(self.adata, method="mtmfft", taper="hann",
                                foi=ftmp, select=select)
            assert np.all(spec.freq == foi)

            # unsorted, duplicate entries in `foi` - result must stay the same
            ftmp = np.hstack([foi, np.full(20, foi[0])])
            spec = freqanalysis(self.adata, method="mtmfft", taper="hann",
                                foi=ftmp, select=select)
            assert np.all(spec.freq == foi)

    def test_dpss(self):

        for sk, select in enumerate(self.sigdataSelections):
            sel = Selector(self.adata, select)
            chanList = np.arange(self.nChannels)[sel.channel]

            # ensure default setting results in single taper
            spec = freqanalysis(self.adata, method="mtmfft",
                                taper="dpss", select=select)
            assert spec.taper.size == 1
            assert np.unique(spec.taper).size == 1
            assert spec.channel.size == len(chanList)

            # specify tapers
            spec = freqanalysis(self.adata, method="mtmfft", taper="dpss",
                                tapsmofrq=7, keeptapers=True, select=select)
            assert spec.taper.size == 7
            assert spec.channel.size == len(chanList)

        # non-equidistant data w/multiple tapers
        artdata = generate_artifical_data(nTrials=5, nChannels=16,
                                          equidistant=False, inmemory=False)
        timeAxis = artdata.dimord.index("time")
        cfg = StructDict()
        cfg.method = "mtmfft"
        cfg.taper = "dpss"
        cfg.tapsmofrq = 9.3

        # trigger error for non-equidistant trials w/o padding
        cfg.pad = None
        with pytest.raises(SPYValueError):
            spec = freqanalysis(cfg, artdata)

        for sk, select in enumerate(self.artdataSelections):

            # unsorted, w/repetitions, do not pad
            cfg.pop("pad", None)
            if select is not None and "toi" in select.keys():
                select["toi"] = self.seed.choice(artdata.time[0], int(artdata.time[0].size))
                cfg.pad = None
            sel = Selector(artdata, select)
            cfg.select = select

            spec = freqanalysis(cfg, artdata)

            # ensure correctness of padding (respecting min. trial length + time-selection)
            if select is None:
                maxtrlno = np.diff(artdata.sampleinfo).argmax()
                tmp = padding(artdata.trials[maxtrlno], "zero", spec.cfg.pad,
                              spec.cfg.padlength, prepadlength=True)
                nSamples = tmp.shape[timeAxis]
            elif "toi" in select:
                nSamples = len(select["toi"])
            else:
                tsel = artdata.time[sel.trials[0]][sel.time[0]]
                nSamples = _nextpow2(tsel.size)
            freqs = np.arange(0, np.floor(nSamples / 2) + 1) * artdata.samplerate / nSamples
            assert spec.freq.size == freqs.size
            assert np.max(spec.freq - freqs) < self.ftol

        # same + reversed dimensional order in input object
        cfg.data = generate_artifical_data(nTrials=5, nChannels=16,
                                           equidistant=False, inmemory=False,
                                           dimord=AnalogData._defaultDimord[::-1])
        timeAxis = cfg.data.dimord.index("time")
        cfg.output = "abs"
        cfg.keeptapers = True

        for select in self.artdataSelections:

            # unsorted, w/repetitions, do not pad
            cfg.pop("pad", None)
            if select is not None and "toi" in select.keys():
                select["toi"] = self.seed.choice(cfg.data.time[0], int(cfg.data.time[0].size))
                cfg.pad = None
            sel = Selector(cfg.data, select)
            cfg.select = select

            spec = freqanalysis(cfg)

            # ensure correctness of padding (respecting min. trial length + time-selection)
            if select is None:
                maxtrlno = np.diff(cfg.data.sampleinfo).argmax()
                tmp = padding(cfg.data.trials[maxtrlno].T, "zero", spec.cfg.pad,
                              spec.cfg.padlength, prepadlength=True)
                nSamples = tmp.shape[not timeAxis]
            elif "toi" in select:
                nSamples = len(select["toi"])
            else:
                tsel = cfg.data.time[sel.trials[0]][sel.time[0]]
                nSamples = _nextpow2(tsel.size)
            freqs = np.arange(0, np.floor(nSamples / 2) + 1) * cfg.data.samplerate / nSamples
            assert spec.freq.size == freqs.size
            assert np.max(spec.freq - freqs) < self.ftol
            assert spec.taper.size > 1

        # same + overlapping trials
        cfg.data = generate_artifical_data(nTrials=5, nChannels=16,
                                           equidistant=False, inmemory=False,
                                           dimord=AnalogData._defaultDimord[::-1],
                                           overlapping=True)
        timeAxis = cfg.data.dimord.index("time")
        cfg.keeptapers = False

        for select in self.artdataSelections:

            # unsorted, w/repetitions, do not pad
            cfg.pop("pad", None)
            if select is not None and "toi" in select.keys():
                select["toi"] = self.seed.choice(cfg.data.time[0], int(cfg.data.time[0].size))
                cfg.pad = None
            sel = Selector(cfg.data, select)
            cfg.select = select

            spec = freqanalysis(cfg)

            # ensure correctness of padding (respecting min. trial length + time-selection)
            if select is None:
                maxtrlno = np.diff(cfg.data.sampleinfo).argmax()
                tmp = padding(cfg.data.trials[maxtrlno].T, "zero", spec.cfg.pad,
                              spec.cfg.padlength, prepadlength=True)
                nSamples = tmp.shape[not timeAxis]
            elif "toi" in select:
                nSamples = len(select["toi"])
            else:
                tsel = cfg.data.time[sel.trials[0]][sel.time[0]]
                nSamples = _nextpow2(tsel.size)
            freqs = np.arange(0, np.floor(nSamples / 2) + 1) * cfg.data.samplerate / nSamples
            assert spec.freq.size == freqs.size
            assert np.max(spec.freq - freqs) < self.ftol
            assert spec.taper.size == 1

    def test_allocout(self):
        # call ``freqanalysis`` w/pre-allocated output object
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
                     keeptapers=False, output="abs", out=out)
        assert out.sampleinfo.shape == (self.nTrials, 2)
        assert out.taper.size == 1

        # re-use `cfg` from above and additionally throw away `tapers`
        cfg.dataset = self.adata
        cfg.out = SpectralData(dimord=SpectralData._defaultDimord)
        cfg.taper = "dpss"
        cfg.keeptapers = False
        freqanalysis(cfg)
        assert cfg.out.taper.size == 1

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
                                keeptapers=False, output="abs", pad="relative",
                                padlength=npad)
            assert (np.diff(avdata.sampleinfo)[0][0] + npad) / 2 + 1 == spec.freq.size
            del avdata, vdata, dmap, spec
            gc.collect()  # force-garbage-collect object so that tempdir can be closed

    @skip_without_dask
    def test_parallel(self, testcluster):
        # collect all tests of current class and repeat them using dask
        # (skip VirtualData tests since ``wrapper_io`` expects valid headers)
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and attr != "test_parallel")]
        all_tests.remove("test_vdata")
        for test in all_tests:
            getattr(self, test)()

        # now create uniform `cfg` for remaining SLURM tests
        cfg = StructDict()
        cfg.method = "mtmfft"
        cfg.taper = "dpss"
        cfg.tapsmofrq = 9.3

        # no. of HDF5 files that will make up virtual data-set in case of channel-chunking
        chanPerWrkr = 7
        nFiles = self.nTrials * (int(self.nChannels/chanPerWrkr)
                                 + int(self.nChannels % chanPerWrkr > 0))

        # simplest case: equidistant trial spacing, all in memory
        fileCount = [self.nTrials, nFiles]
        artdata = generate_artifical_data(nTrials=self.nTrials, nChannels=self.nChannels,
                                          inmemory=True)
        for k, chan_per_worker in enumerate([None, chanPerWrkr]):
            cfg.chan_per_worker = chan_per_worker
            spec = freqanalysis(artdata, cfg)
            assert spec.data.is_virtual
            assert len(spec.data.virtual_sources()) == fileCount[k]

        # non-equidistant trial spacing
        cfg.keeptapers = False
        artdata = generate_artifical_data(nTrials=self.nTrials, nChannels=self.nChannels,
                                          inmemory=True, equidistant=False)
        timeAxis = artdata.dimord.index("time")
        maxtrlno = np.diff(artdata.sampleinfo).argmax()
        tmp = padding(artdata.trials[maxtrlno], "zero", spec.cfg.pad, 
                      spec.cfg.padlength, prepadlength=True)
        nSamples = tmp.shape[timeAxis]
        freqs = np.arange(0, np.floor(nSamples / 2) + 1) * artdata.samplerate / nSamples
        for k, chan_per_worker in enumerate([None, chanPerWrkr]):
            spec = freqanalysis(artdata, cfg)
            assert spec.freq.size == freqs.size
            assert np.allclose(spec.freq, freqs)
            assert spec.taper.size == 1
            
        # equidistant trial spacing, keep tapers
        cfg.output = "abs"
        cfg.keeptapers = True
        artdata = generate_artifical_data(nTrials=self.nTrials, nChannels=self.nChannels,
                                          inmemory=False)
        for k, chan_per_worker in enumerate([None, chanPerWrkr]):
            spec = freqanalysis(artdata, cfg)
            assert spec.taper.size > 1

        # non-equidistant, overlapping trial spacing, throw away trials and tapers
        cfg.keeptapers = False
        cfg.keeptrials = "no"
        artdata = generate_artifical_data(nTrials=self.nTrials, nChannels=self.nChannels,
                                          inmemory=False, equidistant=False,
                                          overlapping=True)
        spec = freqanalysis(artdata, cfg)
        timeAxis = artdata.dimord.index("time")
        maxtrlno = np.diff(artdata.sampleinfo).argmax()
        tmp = padding(artdata.trials[maxtrlno], "zero", spec.cfg.pad,
                      spec.cfg.padlength, prepadlength=True)
        nSamples = tmp.shape[timeAxis]
        freqs = np.arange(0, np.floor(nSamples / 2) + 1) * artdata.samplerate / nSamples
        assert spec.freq.size == freqs.size
        assert np.allclose(spec.freq, freqs)
        assert spec.taper.size == 1
        assert len(spec.time) == 1
        assert len(spec.time[0]) == 1
        assert spec.data.shape == (1, 1, freqs.size, self.nChannels)

        client.close()

    # FIXME: check polyorder/polyremoval once supported
