# -*- coding: utf-8 -*-
#
# Convert between MNE and Syncopy data structures.
#


import numpy as np
import syncopy as spy
from syncopy.shared.parsers import data_parser
from syncopy.shared.errors import SPYValueError, SPYTypeError

__all__ = [
    "raw_adata_to_mne_raw",
    "raw_mne_to_adata",
    "tldata_to_mne_epochs",
    "mne_epochs_to_tldata",
]


def raw_adata_to_mne_raw(adata):
    """
    Convert raw spy.AnalogData (single-trial data) to an MNE Python RawArray.

    This function requires MNE Python (package 'mne') and will raise an `ImportError` if it is not installed.

    Parameters
    ----------
    adata : `AnalogData` instance, must be single-trial data (no trial definition, or a single trial spanning the full data), as `mne.io.RawArray` does not support trials. Use function `tldata_to_mne_epochs` if you want to convert epoched or time-locked `AnalogData` to MNE Python. WARNING: the trial definition, if any, will be completely ignored during export, and the full data will be exported.

    Returns
    -------
    ar : `mne.io.RawArray` instance
    """

    try:
        import mne
    except ImportError:
        raise ImportError("MNE Python not installed, but package 'mne' is required for this function.")
    data_parser(adata, varname="adata", dataclass="AnalogData")
    if (
        len(adata.trials) > 1
    ):  # Check that we have single-trial data, otherwise our concatination of trials along the time axis will lead to unexpected results in the exported data.
        raise SPYValueError(
            legal="AnalogData instance with no trial definition, or a single trial spanning the full data",
            varname="adata",
            actual=f"AnalogData instance with {len(adata.trials)} trials.",
        )
    info = mne.io.meas_info.create_info(list(adata.channel), adata.samplerate, ch_types="misc")
    offset = adata.trialdefinition[0, 2]  # offset in samples, identical over trials.
    ar = mne.io.RawArray((adata.data[()]).T, info, first_samp=offset)
    return ar


def raw_mne_to_adata(ar):
    """
    Convert MNE python `mne.io.RawArray` to `spy.AnalogData` (single-trial data).

    This function requires MNE Python (package 'mne') and will raise an `ImportError` if it is not installed.

    Parameters
    ----------
    ar : `mne.io.RawArray` instance

    Returns
    -------
    adata : `syncopy.AnalogData` instance, with no trial definition (singl-trial data).
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE Python not installed, but package 'mne' is required for this function.")

    if type(ar) != mne.io.RawArray:
        raise SPYTypeError(ar, varname="ar", expected="mne.io.RawArray")

    adata = spy.AnalogData(data=ar.get_data().T, samplerate=ar.info["sfreq"], channel=ar.ch_names)

    samplerate = ar.info["sfreq"]
    offset = ar.first_samp

    # set offset in trial definition
    nSamples = ar.get_data().shape[1]
    trldef = np.vstack(
        [
            np.arange(0, nSamples, nSamples),
            np.arange(0, nSamples, nSamples) + nSamples,
            np.ones(1) * offset,
        ]
    ).T
    adata.trialdefinition = trldef

    return adata


def tldata_to_mne_epochs(tldata):
    """
    Convert Syncopy timelocked data to MNE Python `mne.EpochsArray`.

    This function requires MNE Python (package 'mne') and will raise an `ImportError` if it is not installed.

    Parameters
    ----------
    tldata : `syncopy.TimeLockData` or `AnalogData` instance that is timelocked. If `AnalogData`, the user must make sure that the data is time-locked, which can be tested via the `is_time_locked` property of `Analogdata`. Use function `raw_adata_to_mne_raw` instead if you want to convert raw data without trials to MNE Python.

    Returns
    -------
    epochs : `mne.EpochsArray` instance
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE Python not installed, but package 'mne' is required for this function.")

    if type(tldata) == spy.AnalogData:
        if not tldata.is_time_locked:
            raise SPYValueError(
                legal="TimeLockData instance, or AnalogData instance with is_time_locked == True",
                varname="tldata",
                actual=f"AnalogData instance with is_time_locked == False",
            )

    info = mne.io.meas_info.create_info(list(tldata.channel), tldata.samplerate, ch_types="misc")

    # for MNE, the data needs to have shape (n_epochs, n_channels, n_times) but our
    # TimeLockData has shape (n_times, n_channels) with trials concatenated along the time axis
    num_trials = len(tldata.trials)
    num_channels = len(tldata.channel)
    trial_len = tldata.trials[0].shape[
        0
    ]  # Known to be identical for all trials to due to is_time_locked() check
    data_with_trial_axis = np.zeros((num_trials, num_channels, trial_len), dtype=tldata.data.dtype)
    for trial_idx in range(len(tldata.trials)):
        data_with_trial_axis[trial_idx, :, :] = tldata.trials[trial_idx].T

    offset = tldata.trialdefinition[0, 2]  # offset in samples, identical over trials.
    tmin = -offset / tldata.samplerate

    ea = mne.EpochsArray(data_with_trial_axis, info, tmin=tmin)
    return ea


def mne_epochs_to_tldata(ea):
    """
    Convert MNE EpochsArray to time-locked Syncopy AnalogData instance.

    This function requires MNE Python (package 'mne') and will raise an `ImportError` if it is not installed.

    Parameters
    ----------
    ea : `mne.EpochsArray` instance

    Returns
    -------
    tldata : `syncopy.AnalogData` instance. The trial definition will be set to the MNE epochs, and it is guranteed that the data is time-locked.
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE Python not installed, but package 'mne' is required for this function.")
    if type(ea) != mne.EpochsArray:
        raise SPYTypeError(ea, varname="ea", expected="mne.EpochsArray")

    # ed.data has shape (n_epochs, n_channels, n_times), convert to spy_data with shape (n_times, n_channels) with epochs concatenated along the time axis
    n_epochs = ea.get_data().shape[0]
    n_channels = ea.get_data().shape[1]
    n_times = ea.get_data().shape[2]
    spy_data = np.zeros((n_times * n_epochs, n_channels), dtype=np.float32)

    for chan_idx in range(n_channels):
        spy_data[:, chan_idx] = ea.get_data()[:, chan_idx, :].flatten()

    samplerate = ea.info["sfreq"]
    tldata = spy.AnalogData(data=spy_data, samplerate=samplerate, channel=ea.ch_names)

    offset = -ea.tmin * samplerate  # offset in samples

    nSamples = n_times
    trldef = np.vstack(
        [
            np.arange(0, nSamples * n_epochs, nSamples),
            np.arange(0, nSamples * n_epochs, nSamples) + nSamples,
            np.ones(n_epochs) * offset,
        ]
    ).T
    tldata.trialdefinition = trldef
    return tldata
