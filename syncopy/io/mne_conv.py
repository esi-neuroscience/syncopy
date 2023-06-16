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
    """Convert raw spy.AnalogData (single-trial data) to an MNE RawArray."""

    try:
        import mne
    except ImportError:
        raise ImportError("MNE Python not installed, but package 'mne' is required for this function.")
    data_parser(adata, varname="adata", dataclass="AnalogData")
    info = mne.io.meas_info.create_info(list(adata.channel), adata.samplerate, ch_types='misc')
    ar = mne.io.RawArray((adata.data[()]).T, info)
    return ar


def raw_mne_to_adata(ar):
    """Convert MNE RawArray to spy.AnalogData (single-trial data)."""
    try:
        import mne
    except ImportError:
        raise ImportError("MNE Python not installed, but package 'mne' is required for this function.")
    
    assert type(ar) == mne.io.RawArray, "Invalid input: ar must be of type mne.io.RawArray."
    adata = spy.AnalogData(data=ar.get_data().T, samplerate=ar.info['sfreq'], channel=ar.ch_names)
    return adata


def tldata_to_mne(tldata):
    """Convert epoched data (technically TimeLockData or a AnalogData with equal-length trials and identical trial offsets)
     to MNE EpochsArray."""
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
    #data_with_trial_axis = np.expand_dims(tldata.data[()], axis=0)
    #data_with_trial_axis.reshape((num_trials, trial_len, len(tldata.channel)))
    for trial_idx in range(len(tldata.trials)):
        print(f"tldata.trials[trial_idx] shape is {tldata.trials[trial_idx].shape}")
        data_with_trial_axis[trial_idx,:,:] = tldata.trials[trial_idx].T

    ea = mne.EpochsArray(data_with_trial_axis, info)
    return ea

