# -*- coding: utf-8 -*-
#
# Helpers to parse show keyword outputs
# to generate correct data, labels etc. for the plots
#

import numpy as np
from syncopy.shared.tools import best_match
from syncopy.shared.errors import SPYWarning


def parse_toi(dataobject, show_kwargs):

    """
    Create the (multiple) time axis belonging to a toi/toilim
    selection

    Parameters
    ----------
    dataobject : one derived from :class:`~syncopy.datatype.base_data`
        Syncopy datatype instance, needs to have a `time` property
    show_kwargs : dict
        The keywords provided to the `show` method
    """

    # right now we have to enforce
    # single trial selection only
    trl = show_kwargs.get('trials', None)
    if not isinstance(trl, int):
        SPYWarning("Please select a single trial for plotting!")
        return

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
        offsets = data_y.max(axis=0) + 1
        offsets = np.r_[0, offsets[1:]]
        data_y += offsets

    return data_y
