# -*- coding: utf-8 -*-
#
# Welch's method for the estimation of power spectra, see doi:10.1109/TAU.1967.1161901.
# Since the major part of the work has already been done by mtmconvolv (obtaining the modified
# periodograms), all that's left is the averaging along these periodograms.
#

import numpy as np


def welch(data_arr, axis=0, average="mean"):
    """
    Welch method backend function, works on a single trial.

    Since the major part of the work has already been done by mtmconvolv
    (obtaining the modified periodograms), all that's left is the
    averaging along these periodograms.

    Parameters
    ----------
    data_arr : (N,) :class:`numpy.ndarray`
        Uniformly sampled multi-channel time-series data
        The 1st dimension is interpreted as the time axis
    axis: int
        The axis along which to average.
    average: str, `{'mean', 'median'}`
        The averaging method to use.
    """
    if average == "mean":
        res = np.mean(data_arr, axis=axis, keepdims=True)
    else:
        res = np.median(data_arr, axis=axis, keepdims=True)

    return res