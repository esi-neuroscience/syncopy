# -*- coding: utf-8 -*-
#
# Helpers  to generate correct data, labels etc. for the plots
# from Syncopy dataypes
#

# Builtin/3rd party package imports
import numpy as np
import re

# Syncopy imports
from syncopy.shared.tools import best_match


def parse_foi(dataobject, show_kwargs):

    """
    Create the frequency axis belonging to a foi/foilim
    selection

    Parameters
    ----------
    dataobject : one derived from :class:`~syncopy.datatype.base_data`
        Syncopy datatype instance, needs to have a `freq` property
    show_kwargs : dict
        The keywords provided to the `show` method
    """

    freq = dataobject.freq
    # cut to foi selection
    foilim = show_kwargs.get('foilim', None)
    if foilim is not None:
        freq, _ = best_match(freq, foilim, span=True)
    # here show is broken atm, issue #240
    foi = show_kwargs.get('foi', None)
    if foi is not None:
        freq, _ = best_match(freq, foi, span=False)

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
        The keywords provided to the `show` method
    """

    time = dataobject.time[trl]
    # cut to time selection
    toilim = show_kwargs.get('toilim', None)
    if toilim is not None:
        time, _ = best_match(time, toilim, span=True)
    # here show is broken atm, issue #240
    toi = show_kwargs.get('toi', None)
    if toi is not None:
        time, _ = best_match(time, toi, span=False)

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
        The keywords provided to the `show` method

    Returns
    -------
    labels : str or list
        Depending on the channel selection returns
        a list of str for multiple channels or a single
        str for a single channel selection.
    """

    chs = show_kwargs.get('channel', None)

    # channel selections only allow for arrays and lists
    if hasattr(chs, '__len__'):
        # either str or int for index
        if isinstance(chs[0], str):
            labels = chs
        else:
            labels = ['channel' + str(i + 1) for i in chs]
    # single channel
    elif isinstance(chs, int):
        labels = dataobject.channel[chs]
    elif isinstance(chs, str):
        labels = chs
    # all channels
    else:
        labels = dataobject.channel

    return labels


def shift_multichan(data_y):

    if data_y.ndim > 1:
        # shift 0-line for next channel
        # above max of prev.
        offsets = data_y.max(axis=0)[:-1]
        # shift even further if next channel
        # dips below 0
        offsets += np.abs(data_y.min(axis=0)[1:])
        offsets = np.cumsum(np.r_[0, offsets] * 1.1)
        data_y += offsets

    return data_y


def get_method(dataobject):

    """
    Returns the method string from
    the log of a Syncopy data object
    """

    # get the method string in a capture group
    pattern = re.compile(r'[\s\w\D]+method = (\w+)')
    match = pattern.match(dataobject._log)
    if match:
        meth_str = match.group(1)
        return meth_str


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
        if ncols == nAx and nAx > 7:
            nAx += 1
    # no elif to capture possibly incremented nAx
    if nAx % 2 == 0 and nAx > 2:
        ncols = int(np.sqrt(nAx))  # this is max pltConfig["mMaxYaxes"]
        nrows = ncols
        while(ncols * nrows < nAx):
            nrows -= 1
            ncols = int(nAx / nrows)
    # just two axes
    elif nAx == 2:
        nrows, ncols = 1, 2

    return nrows, ncols
