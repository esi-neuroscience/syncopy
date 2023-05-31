# -*- coding: utf-8 -*-
#
# SpikeData plotting functions
#
# 1st argument **must** be `data` to revert the (plotting-)selections
#

# Builtin/3rd party package imports
import numpy as np
from numbers import Number

# Syncopy imports
from syncopy import __plt__
from syncopy.shared.errors import SPYWarning, SPYValueError
from syncopy.plotting import _plotting
from syncopy.plotting import helpers as plot_helpers
from syncopy.plotting.config import pltErrMsg, pltConfig


@plot_helpers.revert_selection
def plot_single_trial_SpikeData(data, mode='unit', **show_kwargs):
    """
    Plot a single trial of a SpikeData object as a spike raster plot.
    Use `mode` to index spikes either by channel or unit.

    Parameters
    ----------
    data : :class:`~syncopy.datatype.SpikeData`
    mode : {'unit', 'channel'}
        Plot spikes indexed either by unit or channel
    show_kwargs : :func:`~syncopy.datatype.methods.show.show` arguments

    Returns
    -------
    fig : `matplotlib.figure.Figure` instance (or `None` in case of errors), the plot figure.
    ax  : `matplotlib.axes.Axes` instance (or `None` in case of errors), the plot axes.
    """

    trl_sel = show_kwargs.get('trials')
    if not isinstance(trl_sel, Number) and len(data.trials) > 1:
        SPYWarning("Please select a single trial for plotting.. skipping plot!")
        return None, None

    if mode not in ('channel', 'unit'):
        raise SPYValueError("either 'channel' or 'unit'", 'mode', mode)

    # attach in place selection and get the only trial
    data.selectdata(inplace=True, **show_kwargs)
    trl = next(iter(data.selection.trials))
    offset = data.selection.trialdefinition[0, 2]
    trl_start = data.selection.trialdefinition[0, 0]

    ids, labels = _extract_ids_labels(data, mode, show_kwargs)

    fig, ax = _plotting.mk_line_figax(ylabel=mode)

    id_axis = data.dimord.index(mode)

    plot_single_trial(ax, trl, ids,
                      id_axis,
                      data.samplerate,
                      trl_start - offset)

    # for less than 25 units/channels, plot the labels
    if len(labels) <= 25:
        loc = np.arange(len(labels))
        ax.set_yticks(loc, labels)
        ax.set_ylabel('')
    else:
        ax.set_yticks(())

    fig.tight_layout()
    return fig, ax


@plot_helpers.revert_selection
def plot_multi_trial_SpikeData(data, mode='unit', **show_kwargs):
    """
    Plot a few trials (max. 25) of a SpikeData object as a multi-axis figure of
    spike raster plots. Use `mode` to index spikes either by channel or unit.

    Parameters
    ----------
    data : :class:`~syncopy.datatype.SpikeData`
    mode : {'unit', 'channel'}
        Plot spikes indexed either by unit or channel
    show_kwargs : :func:`~syncopy.datatype.methods.show.show` arguments

    Returns
    -------
    fig : `matplotlib.figure.Figure` instance (or `None` in case of errors), the plot figure.
    ax  : `matplotlib.axes.Axes` instance (or `None` in case of errors), the plot axes.
    """

    # max. number of panels/axes
    nTrials_max = 25

    # attach in place selection
    data.selectdata(inplace=True, **show_kwargs)
    nTrials = len(data.selection.trials)

    if nTrials > nTrials_max:
        msg = (f"Cannot plot {nTrials} trials at once!\n"
               f"Please select maximum {nTrials_max} trials for multipanel plotting.. skipping plot!"
               )
        SPYWarning(msg)
        return None, None

    if mode not in ('channel', 'unit'):
        raise SPYValueError("either 'channel' or 'unit'", 'mode', mode)

    # determine axes layout, prefer columns over rows due to display aspect ratio
    nrows, ncols = plot_helpers.calc_multi_layout(nTrials)

    fig, axes = _plotting.mk_multi_line_figax(nrows, ncols, x_size=2, y_size=1.4)

    ids, labels = _extract_ids_labels(data, mode, show_kwargs)
    id_axis = data.dimord.index(mode)

    for trl_id, ax in zip(data.selection.trial_ids, axes.flatten()):
        trl = data.selection.trials[trl_id]
        trl_start = data.trialdefinition[trl_id][0]
        offset = data.trialdefinition[trl_id][2]
        plot_single_trial(ax, trl, ids,
                          id_axis,
                          data.samplerate,
                          trl_start - offset)

    # for less than 20 channels, plot the labels
    # on the leftmost axes
    loc = np.arange(len(labels))
    for ax in axes[:, 0]:
        if len(labels) <= 20:
            ax.set_yticks(loc, labels)
            ax.set_ylabel('')
        else:
            ax.set_ylabel(mode)
    for ax in axes.flatten():
        if len(labels) > 20:
            ax.set_yticks(())

    fig.tight_layout()
    return fig, axes


def plot_single_trial(ax, trl, ids, id_axis, samplerate, sample_shift):
    """
    Plot a single multi-channel trial of SpikeData as a spike raster plot.

    Parameters
    ----------
    ax : Matplotlib axis object
    trl : np.ndarray
        The single trial to plot as NumPy array
    ids : np.ndarray
        Integer array with the channel/unit ids (integers)
        to plot
    id_axis : int
        Which axis of the triak array to match agains the `ids`
    samplerate : float
        The sampling rate
    sample_shift : int
        Sample number to be subtracted from absolute samples
        to arrive at offset relative spike times
    """


    # collect each channel/unit spike times
    all_spikes = []
    for _id in ids:

        # grab the individual channels by boolean indexing
        # includes all units of the original selection (`show_kwargs`)
        spikes = trl[trl[:, id_axis] == _id][:, 0]  # in absolute samples
        spikes = spikes - sample_shift
        all_spikes.append(spikes / samplerate)

    ax.eventplot(all_spikes, alpha=0.7, lineoffsets=1, linelengths=0.8, color='k')


def _extract_ids_labels(data, mode, show_kwargs):
    """
    Helper to extract the integer ids and labels of
    the units/channels to plot
    """

    if mode == 'channel':
        # that are the integer values encoding the channels in the data
        ids = np.arange(len(data.channel))[data.selection.channel]
        # the associated string labels
        labels = data.channel[ids]

    elif mode == 'unit':

        # unit selections are "special" (see `SpikeData._get_unit`)
        # and we need to digest manually to get the ids
        unit_sel = show_kwargs.get('unit')
        if unit_sel is None:
            unit_sel = slice(None)
        # single number/string
        elif not np.shape(unit_sel):
            if isinstance(unit_sel, str):
                unit_sel = np.where(data.unit == unit_sel)[0][0]
        # sequence
        else:
            if isinstance(unit_sel[0], str):
                unit_sel = [np.where(data.unit == unit)[0][0] for unit in unit_sel]

        # that are the integer values encoding the units in the data
        ids = np.arange(len(data.unit))[unit_sel]
        # the associated string labels
        labels = data.unit[ids]

    return ids, labels
