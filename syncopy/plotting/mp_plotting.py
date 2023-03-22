# -*- coding: utf-8 -*-
#
# The singlepanel plotting functions for Syncopy
# data types
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
def plot_AnalogData(data, shifted=True, **show_kwargs):

    """
    The probably simplest plot, 2d-line
    plots of selected channels, one axis for each channel

    Parameters
    ----------
    data : :class:`~syncopy.datatype.AnalogData`
    show_kwargs : :func:`~syncopy.datatype.methods.show.show` arguments
    """

    if not __plt__:
        SPYWarning(pltErrMsg)
        return

    # right now we have to enforce
    # single trial selection only
    trl = show_kwargs.get('trials', None)
    if not isinstance(trl, Number) and len(data.trials) > 1:
        SPYWarning("Please select a single trial for plotting!")
        return

    # only 1 trial so no explicit selection needed
    elif len(data.trials) == 1:
        trl = 0

    # get the data to plot
    data_x = plot_helpers.parse_toi(data, trl, show_kwargs)
    data_y = data.show(**show_kwargs)
    # 'time' and 'channel' are the only axes
    if data._defaultDimord != data.dimord:
        data_y = data_y.T

    if data_y.size == 0:
        lgl = "Selection with non-zero size"
        act = "got zero samples"
        raise SPYValueError(lgl, varname="show_kwargs", actual=act)

    # multiple channels?
    labels = plot_helpers.parse_channel(data, show_kwargs)
    nAx = 1 if isinstance(labels, str) else len(labels)

    if nAx < 2:
        SPYWarning("Please select at least two channels for a multipanelplot!")
        return

    elif nAx > pltConfig['mMaxAxes']:
        SPYWarning(f"Please select max. {pltConfig['mMaxAxes']} channels for a multipanelplot!")
        return
    else:
        # determine axes layout, prefer columns over rows due to display aspect ratio
        nrows, ncols = plot_helpers.calc_multi_layout(nAx)

    fig, axs = _plotting.mk_multi_line_figax(nrows, ncols)

    for chan_dat, ax, label in zip(data_y.T, axs.flatten(), labels):
        _plotting.plot_lines(ax, data_x, chan_dat,
                             label=label,
                             leg_fontsize=pltConfig['mLegendSize'])

    # delete empty plot due to grid extension
    # because of prime nAx -> can be maximally 1 plot
    if ncols * nrows > nAx:
        axs.flatten()[-1].remove()

    fig.tight_layout()
    return fig, axs


@plot_helpers.revert_selection
def plot_SpectralData(data, **show_kwargs):

    """
    Plot either 2d-line plots in case of
    singleton time axis or image plots
    for time-frequency spectra, one for each channel.

    Parameters
    ----------
    data : :class:`~syncopy.datatype.SpectralData`
    show_kwargs : :func:`~syncopy.datatype.methods.show.show` arguments
    """

    if not __plt__:
        SPYWarning(pltErrMsg)
        return

    # right now we have to enforce
    # single trial selection only
    trl = show_kwargs.get('trials', None)
    if not isinstance(trl, Number) and len(data.trials) > 1:
        SPYWarning("Please select a single trial for plotting!")
        return
    elif len(data.trials) == 1:
        trl = 0

    channels = plot_helpers.parse_channel(data, show_kwargs)
    nAx = 1 if isinstance(channels, str) else len(channels)

    if nAx < 2:
        SPYWarning("Please select at least two channels for a multipanelplot!")
        return
    elif nAx > pltConfig['mMaxAxes']:
        SPYWarning("Please select max. {pltConfig['mMaxAxes']} channels for a multipanelplot!")
        return
    else:
        # determine axes layout, prefer columns over rows due to display aspect ratio
        nrows, ncols = plot_helpers.calc_multi_layout(nAx)

    # -- check if it is a time-frequency spectrum ----------
    is_tf = np.any(np.diff(data.trialdefinition)[:, 0] != 1)
    # ------------------------------------------------------
    if is_tf:
        fig, axs = _plotting.mk_multi_img_figax(nrows, ncols)

        # this could be more elegantly solve by
        # an in-place selection?!
        time = plot_helpers.parse_toi(data, trl, show_kwargs)
        freqs = plot_helpers.parse_foi(data, show_kwargs)

        # dimord is time x freq x channel
        # need freq x time each for plotting
        data_cyx = data.show(**show_kwargs).T
        if data_cyx.size == 0:
            lgl = "Selection with non-zero size"
            act = "got zero samples"
            raise SPYValueError(lgl, varname="show_kwargs", actual=act)

        maxP = data_cyx.max()
        for data_yx, ax, label in zip(data_cyx, axs.flatten(), channels):
            _plotting.plot_tfreq(ax, data_yx, time, freqs, vmax=maxP)
            ax.set_title(label, fontsize=pltConfig['mTitleSize'])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.05)

    # just a line plot
    else:
        msg = False
        if 'toilim' in show_kwargs:
            show_kwargs.pop('toilim')
            msg = True
        if 'toi' in show_kwargs:
            show_kwargs.pop('toi')
            msg = True
        if msg:
            msg = ("Line spectra don't have a time axis, "
                   "ignoring `toi/toilim` selection!")
            SPYWarning(msg)

        # get the data to plot
        data_x = plot_helpers.parse_foi(data, show_kwargs)
        output = plot_helpers.get_output(data, 'freqanalysis')

        # only log10 the absolute squared spectra
        if output == 'pow':
            data_y = np.log10(data.show(**show_kwargs))
            ylabel = 'power (dB)'
        elif output in ['fourier', 'complex']:
            SPYWarning("Can't plot complex valued spectra, choose 'real' or 'imag' as output! Aborting plotting.")
            return
        else:
            data_y = data.show(**show_kwargs)
            ylabel = f'{output}'

        taper_labels = None
        if len(data.taper) != 1:   
            taper = show_kwargs.get('taper')
            # multiple tapers are to be plotted
            if not isinstance(taper, (Number, str)):
                taper_labels = data.taper

        fig, axs = _plotting.mk_multi_line_figax(nrows, ncols, xlabel='frequency (Hz)',
                                                 ylabel=ylabel)

        for chan_dat, ax, label in zip(data_y.T, axs.flatten(), channels):
            if taper_labels is not None:
                _plotting.plot_lines(ax, data_x, chan_dat, label=taper_labels, leg_fontsize=pltConfig['mLegendSize'])
            else:
                _plotting.plot_lines(ax, data_x, chan_dat)
            ax.set_title(label, fontsize=pltConfig['mTitleSize'])

        # delete empty plot due to grid extension
        # because of prime nAx -> can be maximally 1 plot
        if ncols * nrows > nAx:
            axs.flatten()[-1].remove()
        fig.tight_layout()

    return fig, axs


@plot_helpers.revert_selection
def plot_CrossSpectralData(data, **show_kwargs):
    """
    Plot 2d-line plots for the different connectivity measures.

    Parameters
    ----------
    data : :class:`~syncopy.datatype.CrossSpectralData`
    show_kwargs : :func:`~syncopy.datatype.methods.show.show` arguments
    """

    if not __plt__:
        SPYWarning(pltErrMsg)
        return

    # right now we have to enforce
    # single trial selection only
    trl = show_kwargs.get('trials', None)
    if not isinstance(trl, int) and len(data.trials) > 1:
        SPYWarning("Please select a single trial for plotting!")
        return
    elif len(data.trials) == 1:
        trl = 0

    # what channel combination
    if 'channel_i' not in show_kwargs or 'channel_j' not in show_kwargs:
        SPYWarning("Please select a channel combination for plotting!")
        return
    chi, chj = show_kwargs['channel_i'], show_kwargs['channel_j']

    # what data do we have?
    method = plot_helpers.get_method(data, 'connectivityanalysis')
    output = plot_helpers.get_output(data, 'connectivityanalysis')

    if method == 'granger':
        xlabel = 'frequency (Hz)'
        ylabel = 'Granger causality'
        label = rf"channel{chi} $\rightarrow$ channel{chj}"
        data_x = plot_helpers.parse_foi(data, show_kwargs)
    elif method == 'coh':
        xlabel = 'frequency (Hz)'
        ylabel = f'{output} coherence'
        label = rf"channel{chi} - channel{chj}"
        data_x = plot_helpers.parse_foi(data, show_kwargs)
    elif method == 'corr':
        xlabel = 'lag'
        ylabel = 'correlation'
        label = rf"channel{chi} - channel{chj}"
        data_x = plot_helpers.parse_toi(data, show_kwargs)
    # that's all the methods we got so far
    else:
        raise NotImplementedError

    # get the data to plot
    data_y = data.show(**show_kwargs)

    # Create the axes and figure if needed.
    # Persistent axes allow for plotting different
    # channel combinations into the same figure.
    if not hasattr(data, 'ax'):
        fig, data.ax = _plotting.mk_line_figax(xlabel, ylabel)
    _plotting.plot_lines(data.ax, data_x, data_y, label=label)

    return fig, data.ax
