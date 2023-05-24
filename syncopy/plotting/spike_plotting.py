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
def plot_single_trial_SpikeData(data, **show_kwargs):
    """
    Plot a single trial of a SpikeData object as a spike raster plot.
    Refers to `plot_single_trial` after `show_kwargs`
    validation.

    Parameters
    ----------
    data : :class:`~syncopy.datatype.SpikeData`
    show_kwargs : :func:`~syncopy.datatype.methods.show.show` arguments

    Returns
    -------
    fig : `matplotlib.figure.Figure` instance (or `None` in case of errors), the plot figure.
    ax  : `matplotlib.axes.Axes` instance (or `None` in case of errors), the plot axes.
    """

    trl_sel = show_kwargs.get('trials')
    if not isinstance(trl_sel, Number) and len(data.trials) > 1:
        SPYWarning("Please select a single trial for plotting.")
        return None, None

    # attach in place selection and get the only trial
    data.selectdata(inplace=True, **show_kwargs)
    trl = next(iter(data.selection.trials))
    offset = data.selection.trialdefinition[0, 2]
    trl_start = data.selection.trialdefinition[0, 0]

    # that's the integer values encoding the channels in the data
    chan_ids = np.arange(len(data.channel))[data.selection.channel]
    # the associated string labels
    chan_labels = data.channel[chan_ids]

    fig, ax = _plotting.mk_line_figax(ylabel='channel')

    plot_single_trial(ax, trl, chan_ids, data.samplerate, trl_start - offset)

    # for less than 25 channels, plot the labels
    if len(chan_labels) <= 25:
        loc = np.arange(len(chan_labels))
        ax.set_yticks(loc, chan_labels)
        ax.set_ylabel('')
    else:
        ax.set_yticks(())
    fig.tight_layout()

    return fig, ax


@plot_helpers.revert_selection
def plot_multi_trial_SpikeData(data, **show_kwargs):
    """
    Plot a few trials (max. 25) of a SpikeData object as a multi-axis figure of
    spike raster plots. Refers to `plot_single_trial` after `show_kwargs`
    validation.

    Parameters
    ----------
    data : :class:`~syncopy.datatype.SpikeData`
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
        msg = (f"Can not plot {nTrials} trials at once!\n"
               f"Please select maximum {nTrials_max} trials for multipanel plotting.. skipping plot"
               )
        SPYWarning(msg)
        return None, None

    # that's the integer values encoding the channels in the data
    chan_ids = np.arange(len(data.channel))[data.selection.channel]
    # the associated string labels
    chan_labels = data.channel[chan_ids]

    # determine axes layout, prefer columns over rows due to display aspect ratio
    nrows, ncols = plot_helpers.calc_multi_layout(nTrials)

    fig, axes = _plotting.mk_multi_line_figax(nrows, ncols, x_size=2, y_size=1.4)

    for trl_id, ax in zip(data.selection.trial_ids, axes.flatten()):
        trl = data.selection.trials[trl_id]
        trl_start = data.trialdefinition[trl_id][0]
        offset = data.trialdefinition[trl_id][2]
        plot_single_trial(ax, trl, chan_ids, data.samplerate, trl_start - offset)

    # for less than 20 channels, plot the labels
    # on the leftmost axes
    loc = np.arange(len(chan_labels))
    for ax in axes[:, 0]:
        if len(chan_labels) <= 20:
            ax.set_yticks(loc, chan_labels)
            ax.set_ylabel('')
        else:
            ax.set_ylabel('channel')
    for ax in axes.flatten():
        if len(chan_labels) > 20:
            ax.set_yticks(())

    fig.tight_layout()
    return fig, axes


def plot_single_trial(ax, trl, chan_ids, samplerate, sample_shift):
    """
    Plot a single multi-channel trial of SpikeData as a spike raster plot.

    Parameters
    ----------
    ax : Matplotlib axis object
    trl : np.ndarray
        The single trial to plot as NumPy array
    chan_ids : np.ndarray
        Integer array with the channel ids (integers)
        to plot
    samplerate : float
        The sampling rate
    sample_shift : int
        Sample number to be subtracted from absolute samples
        to arrive at a offset relative spike time
    """


    # for itrial,trl in enumerate(data.trials):

    # collect each channel spike times
    chan_spikes = []
    for chan_id in chan_ids:

        # grab the individual channels by boolean indexing
        # includes all units of the original selection (`show_kwargs`)
        spikes = trl[trl[:, 1] == chan_id][:, 0]  # in absolute samples
        spikes = spikes - sample_shift
        chan_spikes.append(spikes / samplerate)

    ax.eventplot(chan_spikes, alpha=0.7, lineoffsets=1, linelengths=0.8)
