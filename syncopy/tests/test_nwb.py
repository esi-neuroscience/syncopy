# -*- coding: utf-8 -*-
#
# Test functionality of SyNCoPy-container I/O routines for NWB format files.
#

# Builtin/3rd party package imports
import os
import tempfile
import pytest
import numpy as np
from glob import glob
import matplotlib.pyplot as ppl

# Local imports
import syncopy as spy
from syncopy.datatype import AnalogData
from syncopy.io import load_nwb
from syncopy.io.nwb import _nwb_copy_pynapple
from syncopy.shared.filetypes import FILE_EXT
from syncopy.synthdata.analog import white_noise
from syncopy.synthdata.spikes import poisson_noise
from syncopy.io.load_nwb import _is_valid_nwb_file
from syncopy.tests.helpers import get_file_from_anywhere
from syncopy import __pynwb__

skip_no_pynwb = pytest.mark.skipif(
    not __pynwb__, reason=f"This test requires the 'pynwb' package to be installed."
)


# Decorator to detect if test data dir is available
on_esi = os.path.isdir("/cs/slurm/syncopy")
skip_no_esi = pytest.mark.skipif(not on_esi, reason="ESI fs not available")

# Decorator for pynapple optional tests
has_pynapple = False
try:
    import pynapple as nap

    has_pynapple = True
except ImportError:
    pass
skip_no_pynapple = pytest.mark.skipif(not has_pynapple, reason="pynapple not installed")


class TestNWBImporter:

    nwb_filename = get_file_from_anywhere(["~/test.nwb", "/cs/slurm/syncopy/NWBdata/test.nwb"])
    nwb_filename2 = get_file_from_anywhere(
        [
            "~/adata_no_trials_64chan.nwb",
            "/cs/slurm/syncopy/NWBdata/adata_no_trials_64chan.nwb",
        ]
    )

    @skip_no_pynwb
    def test_load_nwb_analog(self):
        """Test loading of an NWB file containing acquistion data into a Syncopy AnalogData object."""

        if self.nwb_filename is None:
            pytest.skip("Demo NWB file 'test.nwb' not found on current system.")

        spy_filename = self.nwb_filename.split("/")[-1][:-4] + ".spy"
        out = load_nwb(self.nwb_filename, memuse=2000)
        edata, _, adata2 = list(out.values())

        assert isinstance(adata2, spy.AnalogData)
        assert isinstance(edata, spy.EventData)
        assert np.any(~np.isnan(adata2.data))
        assert np.any(adata2.data != 0)

        snippet = adata2.selectdata(latency=[30, 32])

        snippet.singlepanelplot(latency=[30, 30.3], channel=3)
        ppl.gcf().suptitle("raw data")

        # Bandpass filter
        lfp = spy.preprocessing(snippet, filter_class="but", freq=[10, 100], filter_type="bp", order=8)

        # Downsample
        lfp = spy.resampledata(lfp, resamplefs=2000, method="downsample")
        lfp.info = adata2.info
        lfp.singlepanelplot(channel=3)
        ppl.gcf().suptitle("bp-filtered 10-100Hz and resampled")

        spec = spy.freqanalysis(lfp, foilim=[5, 150])
        spec.singlepanelplot(channel=[1, 3])
        ppl.gcf().suptitle("bp-filtered 10-100Hz and resampled")

        # test save and load
        with tempfile.TemporaryDirectory() as tdir:
            lfp.save(os.path.join(tdir, spy_filename))
            lfp2 = spy.load(os.path.join(tdir, spy_filename))

            assert np.allclose(lfp.data, lfp2.data)

    @skip_no_pynwb
    def test_load_our_exported(self):
        """This fails if the exported file was written with older versions of pynwb or its dependencies.
        If you want to re-export the file with your currently running Syncopy version, set the
        variable do_save_testfile to True in function test_save_nwb_analog_no_trialdef_singlechannel below.
        """
        if self.nwb_filename2 is None:
            pytest.skip("Demo NWB file 'adata_no_trials_64chan.nwb' not found on current system.")
        out = load_nwb(self.nwb_filename2)
        assert len(out.channel) == 64
        assert out.data.shape == (1000, 64)


class TestNWBExporter:

    do_validate_NWB = False

    @skip_no_pynwb
    def test_save_nwb_analog_no_trialdef(self):
        """Test saving to NWB file and re-reading data for AnalogData, without trial definition."""

        numChannels = 64
        adata = white_noise(nTrials=1, nChannels=numChannels, nSamples=1000)

        assert isinstance(adata, spy.AnalogData)
        assert len(adata.channel) == numChannels

        with tempfile.TemporaryDirectory() as tdir:
            outpath = os.path.join(tdir, "test_save_analog2nwb0.nwb")
            adata.save_nwb(outpath=outpath, with_trialdefinition=False)

            adata.save_nwb(outpath="adata_no_trials_64chan.nwb", with_trialdefinition=False)

            if self.do_validate_NWB:
                is_valid, err = _is_valid_nwb_file(outpath)
                assert is_valid, f"Exported NWB file failed validation: {err}"

            data_instances_reread = load_nwb(outpath)
            assert (
                len(list(data_instances_reread.values())) == 1
            ), f"Expected 1 loaded data instance, got {len(list(data_instances_reread.values()))}"
            adata_reread = list(data_instances_reread.values())[0]
            assert isinstance(adata_reread, spy.AnalogData), f"Expected AnalogData, got {type(adata_reread)}"
            assert (
                len(adata_reread.channel) == numChannels
            ), f"Expected {numChannels} channels, got {len(adata_reread.channel)}"
            assert len(adata_reread.trials) == 1
            assert all(
                adata_reread.channel == adata.channel
            )  # Check that channel names are saved and re-read correctly.
            assert np.allclose(adata.data, adata_reread.data)

    @skip_no_pynwb
    def test_save_nwb_analog_no_trialdef_singlechannel(self):
        """Test saving to NWB file and re-reading data for AnalogData, without trial definition.
        This time, only save a single channel, due to issues with pynwb saving/loading under m1macos.
        """

        numChannels = 1
        adata = white_noise(nTrials=1, nChannels=numChannels, nSamples=1000)

        assert isinstance(adata, spy.AnalogData)
        assert len(adata.channel) == numChannels

        with tempfile.TemporaryDirectory() as tdir:

            outpath = os.path.join(tdir, "test_save_analog2nwb0_1chan.nwb")
            adata.save_nwb(outpath=outpath, with_trialdefinition=False)

            do_save_testfile = False  # Exports test file used in other tests. Only set to True if you want to re-export the file.
            if do_save_testfile:
                outpath = os.path.expanduser("~/test_save_analog2nwb0_1chan.nwb")
                adata.save_nwb(outpath=outpath, with_trialdefinition=False)

            if self.do_validate_NWB:
                is_valid, err = _is_valid_nwb_file(outpath)
                assert is_valid, f"Exported NWB file failed validation: {err}"

            data_instances_reread = load_nwb(outpath)
            assert (
                len(list(data_instances_reread.values())) == 1
            ), f"Expected 1 loaded data instance, got {len(list(data_instances_reread.values()))}"
            adata_reread = list(data_instances_reread.values())[0]
            assert isinstance(adata_reread, spy.AnalogData), f"Expected AnalogData, got {type(adata_reread)}"
            assert (
                len(adata_reread.channel) == numChannels
            ), f"Expected {numChannels} channels, got {len(adata_reread.channel)}"
            assert len(adata_reread.trials) == 1
            assert all(
                adata_reread.channel == adata.channel
            )  # Check that channel names are saved and re-read correctly.
            assert np.allclose(adata.data, adata_reread.data)

    @skip_no_pynwb
    def test_save_nwb_analog_with_trialdef(self):
        """Test saving to NWB file and re-reading data for AnalogData with a trial definition."""

        numChannels = 64
        numTrials = 5
        adata = white_noise(nTrials=numTrials, nChannels=numChannels, nSamples=1000)

        assert isinstance(adata, spy.AnalogData)
        assert len(adata.channel) == numChannels
        assert len(adata.trials) == numTrials

        with tempfile.TemporaryDirectory() as tdir:
            outpath = os.path.join(tdir, "test_save_analog2nwb.nwb")
            adata.save_nwb(outpath=outpath)

            if self.do_validate_NWB:
                is_valid, err = _is_valid_nwb_file(outpath)
                assert is_valid, f"Exported NWB file failed validation: {err}"

            data_instances_reread = load_nwb(outpath)
            assert (
                len(list(data_instances_reread.values())) == 1
            ), f"Expected 1 loaded data instance, got {len(list(data_instances_reread.values()))}"
            adata_reread = list(data_instances_reread.values())[0]
            assert isinstance(adata_reread, spy.AnalogData), f"Expected AnalogData, got {type(adata_reread)}"
            assert (
                len(adata_reread.channel) == numChannels
            ), f"Expected {numChannels} channels, got {len(adata_reread.channel)}"
            assert len(adata_reread.trials) == numTrials
            assert all(
                adata_reread.channel == adata.channel
            )  # Check that channel names are saved and re-read correctly.
            assert np.allclose(adata.data, adata_reread.data)

    @skip_no_pynwb
    def test_save_nwb_analog_with_trialdef_as_LFP(self):
        """Test saving to NWB file and re-reading data for AnalogData with a trial definition. Saves as LFP, as opposed to raw data."""

        numChannels = 64
        numTrials = 5
        adata = white_noise(nTrials=numTrials, nChannels=numChannels, nSamples=1000)

        assert isinstance(adata, spy.AnalogData)
        assert len(adata.channel) == numChannels
        assert len(adata.trials) == numTrials

        with tempfile.TemporaryDirectory() as tdir:
            outpath = os.path.join(tdir, "test_save_analog2nwb1.nwb")
            adata.save_nwb(outpath=outpath, is_raw=False)

            if self.do_validate_NWB:
                is_valid, err = _is_valid_nwb_file(outpath)
                assert is_valid, f"Exported NWB file failed validation: {err}"

            data_instances_reread = load_nwb(outpath)
            assert (
                len(list(data_instances_reread.values())) == 1
            ), f"Expected 1 loaded data instance, got {len(list(data_instances_reread.values()))}"
            adata_reread = list(data_instances_reread.values())[0]
            assert isinstance(adata_reread, spy.AnalogData), f"Expected AnalogData, got {type(adata_reread)}"
            assert (
                len(adata_reread.channel) == numChannels
            ), f"Expected {numChannels} channels, got {len(adata_reread.channel)}"
            assert len(adata_reread.trials) == numTrials
            assert all(
                adata_reread.channel == adata.channel
            )  # Check that channel names are saved and re-read correctly.
            assert np.allclose(adata.data, adata_reread.data)

    @skip_no_pynwb
    def test_save_nwb_analog_2(self):
        """Test saving to NWB file and re-reading data for 2x AnalogData."""

        numChannels = 64
        numTrials = 5
        adata = white_noise(nTrials=numTrials, nChannels=numChannels, nSamples=1000)
        adata2 = white_noise(nTrials=numTrials, nChannels=numChannels, nSamples=1000)

        assert isinstance(adata, spy.AnalogData)
        assert len(adata.channel) == numChannels
        assert len(adata.trials) == numTrials

        with tempfile.TemporaryDirectory() as tdir:
            outpath = os.path.join(tdir, "test_save_analog2nwb2.nwb")
            nwbfile = adata.save_nwb(outpath=outpath, is_raw=True)
            adata2.save_nwb(
                outpath=outpath,
                nwbfile=nwbfile,
                with_trialdefinition=False,
                is_raw=False,
            )

            if self.do_validate_NWB:
                is_valid, err = _is_valid_nwb_file(outpath)
                assert is_valid, f"Exported NWB file failed validation: {err}"

            data_instances_reread = load_nwb(outpath)
            assert (
                len(list(data_instances_reread.values())) == 2
            ), f"Expected 2 loaded data instances, got {len(list(data_instances_reread.values()))}"
            adata_reread = list(data_instances_reread.values())[1]
            adata2_reread = list(data_instances_reread.values())[0]
            assert isinstance(adata_reread, spy.AnalogData), f"Expected AnalogData, got {type(adata_reread)}"
            assert (
                len(adata_reread.channel) == numChannels
            ), f"Expected {numChannels} channels, got {len(adata_reread.channel)}"
            assert len(adata_reread.trials) == numTrials
            assert all(
                adata_reread.channel == adata.channel
            )  # Check that channel names are saved and re-read correctly.
            assert np.allclose(adata.data, adata_reread.data)
            assert np.allclose(adata2.data, adata2_reread.data)

    @skip_no_pynwb
    def test_save_nwb_timelock_with_trialdef(self):
        """Test saving to NWB file and re-reading data for TimeLockData with a trial definition.
        Currently, when the file is bering re-read, it results in an AnalogData object, not a TimeLockData object.
        This is fine with us,
        """

        numChannels = 64
        numTrials = 5
        adata = white_noise(nTrials=numTrials, nChannels=numChannels, nSamples=1000)

        # Create TimeLockData object from AnalogData object (note: this method loads all data into memory)
        tldata = spy.TimeLockData(
            adata.data[()],
            samplerate=adata.samplerate,
            channel=adata.channel,
            trialdefinition=adata.trialdefinition,
        )

        assert isinstance(adata, spy.AnalogData)
        assert isinstance(tldata, spy.TimeLockData)
        assert len(adata.channel) == numChannels
        assert len(adata.trials) == numTrials
        assert len(tldata.channel) == numChannels
        assert len(tldata.trials) == numTrials

        with tempfile.TemporaryDirectory() as tdir:
            outpath = os.path.join(tdir, "test_save_timelock2nwb.nwb")
            adata.save_nwb(outpath=outpath)

            if self.do_validate_NWB:
                is_valid, err = _is_valid_nwb_file(outpath)
                assert is_valid, f"Exported NWB file failed validation: {err}"

            data_instances_reread = load_nwb(outpath)
            assert (
                len(list(data_instances_reread.values())) == 1
            ), f"Expected 1 loaded data instance, got {len(list(data_instances_reread.values()))}"
            adata_reread = list(data_instances_reread.values())[0]
            assert isinstance(adata_reread, spy.AnalogData), f"Expected AnalogData, got {type(adata_reread)}"
            assert (
                len(adata_reread.channel) == numChannels
            ), f"Expected {numChannels} channels, got {len(adata_reread.channel)}"
            assert len(adata_reread.trials) == numTrials
            assert all(
                adata_reread.channel == adata.channel
            )  # Check that channel names are saved and re-read correctly.
            assert np.allclose(adata.data, adata_reread.data)

    @skip_no_pynwb
    def test_save_nwb_spikedata(self):
        """Test exporting SpikeData to NWB format.

        The data in the NWB file is arranged in a way that is compatible with Pynapple.
        """
        nTrials = 10
        nSpikes = 20_000
        samplerate = 10_000
        nChannels = 3
        nUnits = 5
        spdata = poisson_noise(
            nTrials=nTrials,
            nSpikes=nSpikes,
            samplerate=samplerate,
            nChannels=nChannels,
            nUnits=nUnits,
        )

        assert isinstance(spdata, spy.SpikeData)

        nChannelsExpectedOnRead = 1  # The NWB format does not store channels for spike data, only units, so we get a single channel back.

        with tempfile.TemporaryDirectory() as tdir:
            nwb_outpath = os.path.join(tdir, "test_save_spike2nwb.nwb")
            spdata.save_nwb(outpath=nwb_outpath)

            if self.do_validate_NWB:
                is_valid, err = _is_valid_nwb_file(nwb_outpath)
                assert is_valid, f"Exported NWB file failed validation: {err}"

            ##  Save another copy to home directory for manual inspection and
            ##   upload to nwbexplorer at http://nwbexplorer.opensourcebrain.org
            # from os.path import expanduser
            # spdata.save_nwb(outpath=os.path.join(expanduser("~"), 'spikes.nwb'))

            data_instances_reread = load_nwb(nwb_outpath)
            assert (
                len(list(data_instances_reread.values())) == 1
            ), f"Expected 1 loaded data instance, got {len(list(data_instances_reread.values()))}"
            spdata_reread = list(data_instances_reread.values())[0]
            assert isinstance(spdata_reread, spy.SpikeData), f"Expected SpikeData, got {type(spdata_reread)}"
            assert (
                len(spdata_reread.channel) == nChannelsExpectedOnRead
            ), f"Expected {nChannelsExpectedOnRead} channels, got {len(spdata_reread.channel)}"
            assert len(spdata_reread.trials) == nTrials
            assert spdata_reread.samplerate == samplerate

            assert (
                spdata.data.shape == spdata_reread.data.shape
            ), f"Expected identical shapes, got original={spdata.data.shape}, reread={spdata_reread.data.shape}"
            assert np.allclose(spdata.data[:, 0], spdata_reread.data[:, 0])

    @skip_no_pynapple
    @skip_no_pynwb
    def test_load_exported_nwb_spikes_pynapple(self, plot_spikes=True):
        """Test loading exported SpikeData in pynapple.

        Also demonstrates how to use the pynapple API to access trials
        and plot some spike times.
        """

        spdata = poisson_noise()

        if plot_spikes:
            import matplotlib.pyplot as plt

            plt.ion()
            spdata.multipanelplot(unit=np.arange(10), on_yaxis="unit", trials=0)

        import pynapple as pna

        with tempfile.TemporaryDirectory() as tdir:
            nwb_outpath = os.path.join(tdir, "test_spike2nwb_pynapple.nwb")
            spdata.save_nwb(outpath=nwb_outpath)

            _nwb_copy_pynapple(nwb_outpath, tdir)
            pyndata = nap.load_session(tdir, "neurosuite")
            assert len(pyndata.epochs) == 10  # trials
            spikes = pyndata.spikes
            assert (
                type(spikes) == pna.core.ts_group.TsGroup
            ), f"Expected list of SpikeData, got {type(spikes)}"
            neuron_0 = spikes[0]
            assert hasattr(neuron_0, "times")
            assert hasattr(pyndata, "epochs")

            spikes_trial0 = pyndata.spikes.restrict(
                pyndata.epochs.get("trial 0")
            )  # select trial 0 in Pynapple.
            spikes = spikes_trial0

            if plot_spikes:
                import matplotlib.gridspec as gridspec

                unit_count = len(spikes)
                assert unit_count == 10, f"Expected 10 units, got {unit_count}"
                gs = gridspec.GridSpec(unit_count, 1)
                plt.figure()
                for neuron_idx, ts in spikes.items():
                    ax = plt.subplot(gs[neuron_idx, 0])
                    if neuron_idx == 0:
                        ax.set_title("Spike times in Pynapple")
                    assert hasattr(ts, "times"), f"Neuron #{ts} with has no times attribute: {type(ts)}"
                    assert (
                        type(ts.times()) == np.ndarray
                    ), f"Expected list of SpikeData, got {type(ts.times())}"
                    ax.vlines(ts.times(), 0, 1)
                    ax.get_xaxis().set_label_text("Time (s)")
                    ax.set_ylabel(f"Unit {neuron_idx+1}")
                plt.show()


if __name__ == "__main__":
    T0 = TestNWBImporter()
    T1 = TestNWBExporter()
