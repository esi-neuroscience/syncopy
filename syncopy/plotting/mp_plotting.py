# -*- coding: utf-8 -*-
#
# The singlepanel plotting functions for Syncopy
# data types
#

# Builtin/3rd party package imports
import numpy as np

# Syncopy imports
from syncopy import __plt__
from syncopy.shared.errors import SPYWarning
from syncopy.plotting import _plotting
from syncopy.plotting import _helpers as plot_helpers
from syncopy.plotting.config import pltErrMsg, pltConfig


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
    if not isinstance(trl, int) and len(data.trials) > 1:
        SPYWarning("Please select a single trial for plotting!")
        return
    # only 1 trial so no explicit selection needed
    elif len(data.trials) == 1:
        trl = 0

    # get the data to plot
    data_x = plot_helpers.parse_toi(data, trl, show_kwargs)
    data_y = data.show(**show_kwargs)

    # multiple channels?
    labels = plot_helpers.parse_channel(data, show_kwargs)
    nAx = 1 if isinstance(labels, str) else len(labels)

    if nAx < 2:
        SPYWarning("Please select at least two channels for a multipanelplot!")
        return
    elif nAx > pltConfig['mMaxAxes']:
        SPYWarning("Please select max. {pltConfig['mMaxAxes']} channels for a multipanelplot!")
        return
    else:
        # determine axes layout, prefer columns over rows due to display aspect ratio
        nrows, ncols = plot_helpers.calc_multi_layout(nAx)

    fig, axs = _plotting.mk_multi_line_figax(nrows, ncols)

    for chan_dat, ax, label in zip(data_y.T, axs.flatten(), labels):
        _plotting.plot_lines(ax, data_x, chan_dat, label=label, leg_fontsize=pltConfig['mLegendSize'])

    # delete empty plot due to grid extension
    # because of prime nAx -> can be maximally 1 plot
    if ncols * nrows > nAx:
        axs.flatten()[-1].remove()

    fig.tight_layout()


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
    if not isinstance(trl, int) and len(data.trials) > 1:
        SPYWarning("Please select a single trial for plotting!")
        return
    elif len(data.trials) == 1:
        trl = 0

    labels = plot_helpers.parse_channel(data, show_kwargs)
    nAx = 1 if isinstance(labels, str) else len(labels)

    if nAx < 2:
        SPYWarning("Please select at least two channels for a multipanelplot!")
        return
    elif nAx > pltConfig['mMaxAxes']:
        SPYWarning("Please select max. {pltConfig['mMaxAxes']} channels for a multipanelplot!")
        return
    else:
        # determine axes layout, prefer columns over rows due to display aspect ratio
        nrows, ncols = plot_helpers.calc_multi_layout(nAx)

    # how got the spectrum computed
    method = plot_helpers.get_method(data)
    if method in ('wavelet', 'superlet', 'mtmconvol'):        
        fig, axs = _plotting.mk_multi_img_figax(nrows, ncols)

        time = plot_helpers.parse_toi(data, trl, show_kwargs)
        # dimord is time x freq x channel
        # need freq x time each for plotting
        data_cyx = data.show(**show_kwargs).T
        maxP = data_cyx.max()
        for data_yx, ax, label in zip(data_cyx, axs.flatten(), labels):
            _plotting.plot_tfreq(ax, data_yx, time, data.freq, vmax=maxP)
            ax.set_title(label, fontsize=pltConfig['mTitleSize'])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.05)

    # just a line plot
    else:
        # get the data to plot
        data_x = plot_helpers.parse_foi(data, show_kwargs)
        data_y = np.log10(data.show(**show_kwargs))

        fig, axs = _plotting.mk_multi_line_figax(nrows, ncols, xlabel='frequency (Hz)',
                                                 ylabel='power (dB)')

        for chan_dat, ax, label in zip(data_y.T, axs.flatten(), labels):
            _plotting.plot_lines(ax, data_x, chan_dat, label=label, leg_fontsize=pltConfig['mLegendSize'])

        # delete empty plot due to grid extension
        # because of prime nAx -> can be maximally 1 plot
        if ncols * nrows > nAx:
            axs.flatten()[-1].remove()
        fig.tight_layout()


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
    method = plot_helpers.get_method(data)
    if method == 'granger':
        xlabel = 'frequency (Hz)'
        ylabel = 'Granger causality'
        label = rf"channel{chi} $\rightarrow$ channel{chj}"
        data_x = plot_helpers.parse_foi(data, show_kwargs)
    elif method == 'coh':
        xlabel = 'frequency (Hz)'
        ylabel = 'coherence'
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

    # create the axes and figure if needed
    # persisten axes allows for plotting different
    # channel combinations into the same figure
    if not hasattr(data, 'ax'):
        fig, data.ax = _plotting.mk_line_figax(xlabel, ylabel)
    _plotting.plot_lines(data.ax, data_x, data_y, label=label)
