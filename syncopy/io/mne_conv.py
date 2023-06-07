# -*- coding: utf-8 -*-
#
# Convert between MNE and Syncopy data structures.
#

import numpy as np
import syncopy as spy

# See https://github.com/mne-tools/mne-python/blob/maint/1.4/mne/io/fieldtrip/fieldtrip.py
# for how MNE handles FieldTrip data structures.

__all__ = ["raw_adata_to_mne"]


def raw_adata_to_mne(adata):
    """Convert raw spy.AnalogData (single-trial data) to an MNE RawArray."""

    try:
        import mne
    except ImportError:
        raise ImportError("MNE Python not installed, but package 'mne' is required for this function.")
    info = mne.io.meas_info.create_info(list(adata.channel), adata.samplerate, ch_types='misc')
    ar = mne.io.RawArray((adata.data[()]).T, info)
    return ar
