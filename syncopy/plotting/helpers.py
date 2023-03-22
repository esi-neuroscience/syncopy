# -*- coding: utf-8 -*-
#
# Helpers  to generate correct data, labels etc. for the plots
# from Syncopy dataypes
#

# Builtin/3rd party package imports
import numpy as np
from copy import deepcopy
import re
import functools


def revert_selection(plotter):

    """
    To extract 'meta-information' like time and freq axis
    for a particular plot we use (implicit from the users
    perspective) selections. To return to a clean slate
    we revert/delete it afterwards.

    All plotting routines must have `data` as 1st argument!
    """
    @functools.wraps(plotter)
    def wrapper_plot(data, *args, **kwargs):

        # to restore
        select_backup = None if data.selection is None else deepcopy(data.selection.select)

        res = plotter(data, *args, **kwargs)

        # restore initial selection or wipe
        if select_backup:
            data.selectdata(select_backup, inplace=True)
        else:
            data.selection = None

        return res

    return wrapper_plot


def parse_foi(dataobject, show_kwargs):

    """
    Create the frequency axis belonging to a foi/foilim
    selection

    Parameters
    ----------
    dataobject : one derived from :class:`~syncopy.datatype.base_data`
        Syncopy datatype instance, needs to have a `freq` property
    show_kwargs : dict
        The keywords provided to the `selectdata` method
    """

    # apply the selection
    dataobject.selectdata(inplace=True, **show_kwargs)

    idx = dataobject.selection.freq
    # index selection, only one `freq` for all trials
    freq = dataobject.freq[idx]

    return freq


def parse_toi(dataobject, trl, show_kwargs):

    """
    Create the (multiple) time axis belonging to a toi/toilim
    selection

    Parameters
    ----------
    dataobject : one derived from :class:`~syncopy.datatype.base_data`
        Syncopy datatype instance, needs to have a `time` property
    trl : int
        The index of the selected trial to plot
    show_kwargs : dict
        The keywords provided to the `selectdata` method
    """

    # apply the selection
    dataobject.selectdata(inplace=True, **show_kwargs)

    # still have to index the only and single trial
    idx = dataobject.selection.time[0]

    # index selection, again the single trial
    time = dataobject.time[trl][idx]

    return time


def parse_channel(dataobject, show_kwargs):

    """
    Create the labels from a channel
    selection

    Parameters
    ----------
    dataobject : one derived from :class:`~syncopy.datatype.base_data`
        Syncopy datatype instance, needs to have a `channel` property
    show_kwargs : dict
        The keywords provided to the `selectdata` method

    Returns
    -------
    labels : str or list
        Depending on the channel selection returns
        a list of str for multiple channels or a single
        str for a single channel selection.
    """

    # apply selection
    dataobject.selectdata(inplace=True, **show_kwargs)

    # get channel labels
    idx = dataobject.selection.channel
    labels = dataobject.channel[idx]

    # make sure a single string is returned
    # if only one channel is selected
    if np.size(labels) == 1 and np.ndim(labels) != 0:
        labels = labels[0]

    return labels


def shift_multichan(data_y):

    if data_y.ndim > 1:
        # shift 0-line for next channel
        # above max of prev- min of current
        offsets = data_y.max(axis=0)[:-1]
        offsets -= data_y.min(axis=0)[1:]
        offsets = np.cumsum(np.r_[0, offsets] * 1.1)
    else:
        offsets = 0

    return offsets


def get_method(dataobject, frontend_name):

    """
    Returns the method string from
    the cfg of a Syncopy data object
    """

    cfg_entry = dataobject.cfg[frontend_name]
    return cfg_entry.get('method')


def get_output(dataobject, frontend_name):

    """
    Returns the output string from
    the cfg of a Syncopy data object
    """

    cfg_entry = dataobject.cfg[frontend_name]
    return cfg_entry.get('output')


def calc_multi_layout(nAx):

    """
    Given the total numbers of
    axes `nAx` create the nrows, ncols
    layout. In case of `nAx` being prime,
    augment by 1 to rather leave an
    empty plot than to have say a (17, 1) layout..
    """

    # This works as well as long
    # as `nAx` isn't prime :]
    # so we have to augment by 1 if that's the case
    if nAx % 2 != 0:
        ncols = int(np.sqrt(nAx))  # this is max pltConfig["mMaxYaxes"]
        nrows = ncols
        while(ncols * nrows < nAx):
            ncols += 1
            nrows = int(nAx / ncols)
        # nAx was prime and too big
        # for one plotting row
        if ncols == nAx and nAx > 4:
            nAx += 1
    # no elif to capture possibly incremented nAx
    if nAx % 2 == 0 and nAx > 2:
        nrows = int(np.sqrt(nAx))  # this is max pltConfig["mMaxYaxes"]
        ncols = nAx // nrows
        while(ncols * nrows < nAx):
            nrows -= 1
            ncols = int(nAx / nrows)
    # just two axes
    elif nAx == 2:
        nrows, ncols = 1, 2

    return nrows, ncols


def check_if_time_freq(data):
    """
    Looks into the first column of the trialdefinition
    to determine if there is a real time axis, or it is
    just trial stacking.
    """
    is_tf = np.any(np.diff(data.trialdefinition)[:, 0] != 1)

    return is_tf
