# -*- coding: utf-8 -*-
#
# Convert between MNE and Syncopy data structures.
#


import numpy as np
import syncopy as spy
from syncopy.shared.parsers import data_parser
# See https://github.com/mne-tools/mne-python/blob/maint/1.4/mne/io/fieldtrip/fieldtrip.py
# for how MNE handles FieldTrip data structures.

__all__ = ["raw_adata_to_mne", "raw_mne_to_adata", "tldata_to_mne"]


def raw_adata_to_mne(adata):
    """
    Convert raw spy.AnalogData (single-trial data) to an MNE RawArray.

    This function requires MNE Python (package 'mne') and will raise an ImportError if it is not installed.

    Parameters
    ----------
    adata : AnalogData, must be single-trial data (no trial definition), as MNE RawArray does not support trials. Use function `tldata_to_mne` if you want to convert epoched or time-locked AnalogData to MNE.

    Returns
    -------
    ar : mne.io.RawArray
    """

    try:
        import mne
    except ImportError:
        raise ImportError("MNE Python not installed, but package 'mne' is required for this function.")
    data_parser(adata, varname="adata", dataclass="AnalogData")
    info = mne.io.meas_info.create_info(list(adata.channel), adata.samplerate, ch_types='misc')
    ar = mne.io.RawArray((adata.data[()]).T, info)
    return ar


def raw_mne_to_adata(ar):
    """
    Convert MNE RawArray to spy.AnalogData (single-trial data).

    This function requires MNE Python (package 'mne') and will raise an ImportError if it is not installed.

    Parameters
    ----------
    ar : mne.io.RawArray

    Returns
    -------
    adata : AnalogData
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE Python not installed, but package 'mne' is required for this function.")
    
    assert type(ar) == mne.io.RawArray, "Invalid input: ar must be of type mne.io.RawArray."
    adata = spy.AnalogData(data=ar.get_data().T, samplerate=ar.info['sfreq'], channel=ar.ch_names)
    return adata


def tldata_to_mne(tldata):
    """
    Convert timelocked data to MNE EpochsArray.

    This function requires MNE Python (package 'mne') and will raise an ImportError if it is not installed.

    Parameters
    ----------
    tldata : TimeLockData or AnalogData that is timelocked. If AnalogData, the user must make sure that the data is time-locked, which can be tested via the `is_time_locked` property of `Analogdata`. Use function `raw_adata_to_mne` instead if you want to convert raw data without trials to MNE.

    Returns
    -------
    epochs : mne.EpochsArray
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE Python not installed, but package 'mne' is required for this function.")
    assert type(tldata) == spy.AnalogData or type(tldata) == spy.TimeLockData, "Invalid input: tldata must be of type AnalogData or TimeLockData."
    if type(tldata) == spy.AnalogData:
        assert tldata.is_time_locked == True, "Invalid input: tldata must be time-locked."
    info = mne.io.meas_info.create_info(list(tldata.channel), tldata.samplerate, ch_types='misc')

    # for MNE, the data needs to have shape (n_epochs, n_channels, n_times) but our
    # TimeLockData has shape (n_times, n_channels) with trials concatenated along the time axis
    num_trials = len(tldata.trials)
    num_channels = len(tldata.channel)
    trial_len = tldata.trials[0].shape[0]  # Known to be identical for all trials to due to is_time_locked() check
    data_with_trial_axis = np.zeros((num_trials, num_channels, trial_len), dtype=tldata.data.dtype)
    for trial_idx in range(len(tldata.trials)):
        data_with_trial_axis[trial_idx,:,:] = tldata.trials[trial_idx].T

    ea = mne.EpochsArray(data_with_trial_axis, info)
    return ea

def mne_epochs_to_tldata(ea):
    """
    Convert MNE EpochsArray to TimeLockData.

    This function requires MNE Python (package 'mne') and will raise an ImportError if it is not installed.

    Parameters
    ----------
    ea : mne.EpochsArray

    Returns
    -------
    tldata : TimeLockData
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE Python not installed, but package 'mne' is required for this function.")
    assert type(ea) == mne.EpochsArray, "Invalid input: ea must be of type mne.EpochsArray."
    # ed.data has shape (n_epochs, n_channels, n_times), convert to spy_data with shape (n_times, n_channels) with epochs concatenated along the time axis
    n_times = ea.get_data().shape[2]
    n_epochs = ea.get_data().shape[0]
    n_channels = ea.get_data().shape[1]
    spy_data = np.zeros((n_times * n_epochs, n_channels), dtype=np.float32)
    for chan_idx in range(n_epochs):
        spy_data[:,chan_idx] = ea.get_data()[:,chan_idx,:].flatten()

    tldata = spy.AnalogData(data=spy_data, samplerate=ea.info['sfreq'], channel=ea.ch_names)
    return tldata