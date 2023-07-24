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
from syncopy.shared.errors import SPYWarning, SPYValueError, SPYError
from syncopy.plotting import _plotting
from syncopy.plotting import helpers as plot_helpers
from syncopy.plotting.config import pltErrMsg, pltConfig


@plot_helpers.revert_selection
def plot_single_figure_SpikeData(data, on_yaxis="trials", **show_kwargs):
    """
    Plot a single unit/trial of a SpikeData object as a spike raster plot.
    Use `on_yaxis` to index spikes either by trials, unit or channel.

    Parameters
    ----------
    data : :class:`~syncopy.datatype.SpikeData`
    on_yaxis : {'trials', 'unit', 'channel'}
        Plot a spike rasterplot with either unit, channel or trials on the y-axis.
        In case of `'trials'`, need to select a single unit with `**show_kwargs`
    show_kwargs : :func:`~syncopy.datatype.methods.show.show` arguments

    Returns
    -------
    fig : `matplotlib.figure.Figure` instance (or `None` in case of errors), the plot figure.
    ax  : `matplotlib.axes.Axes` instance (or `None` in case of errors), the plot axes.
    """

    trl_sel = show_kwargs.get("trials")

    if on_yaxis not in ("channel", "unit", "trials"):
        raise SPYValueError("either 'trials', 'unit' or 'channel'", "on_yaxis", on_yaxis)

    # attach in place selection and get the only trial
    data.selectdata(inplace=True, **show_kwargs)
    trl = next(iter(data.selection.trials))
    offset = data.selection.trialdefinition[0, 2]
    trl_start = data.selection.trialdefinition[0, 0]

    fig, ax = _plotting.mk_line_figax(ylabel=on_yaxis)

    if on_yaxis != "trials":

        if not isinstance(trl_sel, Number) and len(data.trials) > 1:
            raise SPYError("Please select a single trial for plotting!")

        ids, labels = _extract_ids_labels(data, on_yaxis, show_kwargs)
        id_axis = data.dimord.index(on_yaxis)

        _plot_single_trial(ax, trl, ids, id_axis, data.samplerate, trl_start - offset)

    # single unit on trial y-axis
    else:
        # still need to get the single unit label
        ids, unit_label = _extract_ids_labels(data, "unit", show_kwargs)
        if np.array(ids).size != 1:
            raise SPYError("Please select a single unit for plotting!")

        labels = ["trial" + str(trl_id) for trl_id in data.selection.trial_ids]

        _plot_single_unit(ax, data, int(ids), data.samplerate)
        ax.set_title(unit_label)
    # for less than 25 units/channels, plot all the labels
    if len(labels) <= 25:
        loc = np.arange(len(labels))
        ax.set_yticks(loc, labels)
        ax.set_ylabel("")

    fig.tight_layout()
    return fig, ax


@plot_helpers.revert_selection
def plot_multi_figure_SpikeData(data, on_yaxis="trials", **show_kwargs):
    """
    Plot a few trials (max. 25) of a SpikeData object as a multi-axis figure of
    spike raster plots. Use `on_yaxis` to index spikes either by channel or unit.

    Parameters
    ----------
    data : :class:`~syncopy.datatype.SpikeData`
    on_yaxis : {'trials', 'unit', 'channel'}
        Plot a spike rasterplot with either unit, channel or trials on the y-axis.
        In case of `'trials'`, need to select at max. 25 units with `**show_kwargs`
    show_kwargs : :func:`~syncopy.datatype.methods.show.show` arguments

    Returns
    -------
    fig : `matplotlib.figure.Figure` instance (or `None` in case of errors), the plot figure.
    ax  : `matplotlib.axes.Axes` instance (or `None` in case of errors), the plot axes.
    """

    # max. number of panels/axes
    subplots_max = 25

    # attach in place selection
    data.selectdata(inplace=True, **show_kwargs)
    nTrials = len(data.selection.trials)

    if on_yaxis not in ("channel", "unit", "trials"):
        raise SPYValueError("either 'trials', 'unit' or 'channel'", "on_yaxis", on_yaxis)

    if on_yaxis != "trials":

        if nTrials > subplots_max:
            msg = (
                f"Cannot plot {nTrials} trials at once!\n"
                f"Please select maximum {subplots_max} trials for multipanel plotting.. skipping plot!"
            )
            raise SPYError(msg)

        # determine axes layout, prefer columns over rows due to common landscape display aspect ratio
        nrows, ncols = plot_helpers.calc_multi_layout(nTrials)

        ids, labels = _extract_ids_labels(data, on_yaxis, show_kwargs)
        id_axis = data.dimord.index(on_yaxis)

    # trial y-axis
    else:
        # still need to get the unit labels, id
        ids, labels = _extract_ids_labels(data, "unit", show_kwargs)
        nUnits = np.array(ids).size
        if nUnits > subplots_max:
            msg = (
                f"Cannot plot {nUnits} units at once!\n"
                f"Please select maximum {subplots_max} units for multipanel plotting.. skipping plot!"
            )
            raise SPYError(msg)

        # determine axes layout, prefer columns over rows due to common landscape display aspect ratio
        nrows, ncols = plot_helpers.calc_multi_layout(nUnits)

    fig, axes = _plotting.mk_multi_line_figax(nrows, ncols, x_size=2, y_size=1.4)

    if on_yaxis != "trials":

        for trl_id, ax in zip(data.selection.trial_ids, axes.flatten()):
            trl = data.selection.trials[trl_id]
            trl_start = data.trialdefinition[trl_id][0]
            offset = data.trialdefinition[trl_id][2]
            _plot_single_trial(ax, trl, ids, id_axis, data.samplerate, trl_start - offset)
            ax.set_title("trial" + str(trl_id))

    # trial on y-axis
    else:
        for unit_id, unit_label, ax in zip(ids, labels, axes.flatten()):
            _plot_single_unit(ax, data, unit_id, data.samplerate)
            ax.set_title(unit_label)
        # overwrite labels
        labels = ["trial" + str(trl_id) for trl_id in data.selection.trial_ids]

    # for less than 25 labels, plot all the labels
    # on the leftmost axes
    loc = np.arange(len(labels))
    for ax in axes[:, 0]:
        if len(labels) <= 25:
            ax.set_yticks(loc, labels)
            ax.set_ylabel("")
        else:
            ax.set_ylabel(on_yaxis)

    fig.tight_layout()
    return fig, axes


def _plot_single_trial(ax, trl, ids, id_axis, samplerate, sample_shift):
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

    ax.eventplot(all_spikes, alpha=0.5, lineoffsets=1, linelengths=0.8, color="k")

    # add 0 line if pre-stim is available
    if np.any(ax.get_xticks() < 0):
        ax.plot([0, 0], [-0.5, len(ids) - 0.5], "r--", lw=1.3)


def _plot_single_unit(ax, data, unit_id, samplerate):
    """
    Plot a single multi-channel trial of SpikeData as a spike raster plot.

    Parameters
    ----------
    ax : Matplotlib axis object
    data : :class:`~syncopy.datatype.SpikeData`
    unit_id : int
        Integer encoding the single unit to plot
    samplerate : float
        The sampling rate
    """

    # collect each channel/unit spike times
    all_spikes = []
    for i, trl in enumerate(data.selection.trials):

        offset = data.selection.trialdefinition[i, 2]
        trl_start = data.selection.trialdefinition[i, 0]
        sample_shift = trl_start - offset

        # grab the selected unit for each trial
        spikes = trl[trl[:, 2] == unit_id][:, 0]  # in absolute samples
        spikes = spikes - sample_shift
        all_spikes.append(spikes / samplerate)

    ax.eventplot(all_spikes, alpha=0.5, lineoffsets=1, linelengths=0.8, color="k")

    # add 0 line if pre-stim is available
    if np.any(ax.get_xticks() < 0):
        ax.plot([0, 0], [-0.5, len(data.selection.trials) - 0.5], "r--", lw=1.3)


def _extract_ids_labels(data, on_yaxis, show_kwargs):
    """
    Helper to extract the integer ids and labels of
    the units/channels to plot
    """

    if on_yaxis == "channel":
        # that are the integer values encoding the channels in the data
        ids = np.arange(len(data.channel))[data.selection.channel]
        # the associated string labels
        labels = data.channel[ids]

    elif on_yaxis == "unit":

        # unit selections are "special" (see `SpikeData._get_unit`)
        # and we need to digest manually to get the ids
        unit_sel = show_kwargs.get("unit")
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
