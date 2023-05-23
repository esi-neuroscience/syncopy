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
    trl = next(iter(data.trials))

    # that's the integer value encoding a channel in the data
    chan_ids = data.selection.channel

    # the associated string labels
    chan_labels = data.channel[chan_ids]

    fig, ax = _plotting.mk_line_figax(ylabel='channel')

    plot_single_trial(ax, trl, chan_ids, data.samplerate)

    # for less than 25 channels, plot the labels
    if len(chan_labels) <= 25:
        loc = np.arange(len(chan_labels))
        ax.set_yticks(loc, chan_labels)
        ax.set_ylabel('')
    else:
        ax.set_yticks(())
    fig.tight_layout()


def plot_single_trial(ax, trl, chan_ids, samplerate):
    """
    Plot a single multi-channel trial of SpikeData as a spike raster plot.

    Parameters
    ----------

    """


    # for itrial,trl in enumerate(data.trials):

    # collect each channel spike times
    chan_spikes = []
    for chan_id in chan_ids:

        # grab the individual channels by boolean indexing
        # includes all units of the original selection (`show_kwargs`)
        spikes = trl[trl[:, 1] == chan_id][:, 0]
        chan_spikes.append(spikes / samplerate)

    ax.eventplot(chan_spikes, alpha=0.7, lineoffsets=1, linelengths=1)
