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
from syncopy.plotting import _helpers as plot_helpers
from syncopy.plotting.config import pltErrMsg, pltConfig


@plot_helpers.revert_selection
def plot_AnalogData(data, shifted=True, **show_kwargs):

    """
    The probably simplest plot, a 2d-line
    plot of selected channels

    Parameters
    ----------
    data : :class:`~syncopy.datatype.AnalogData`
    show_kwargs : :func:`~syncopy.datatype.methods.show.show` arguments

    Returns
    -------
    fig : `~matplotlib.figure.Figure`

    ax : `~matplotlib.axes.Axes`
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
    data_y = data.show(**show_kwargs)
    if data_y.size == 0:
        lgl = "Selection with non-zero size"
        act = "got zero samples"
        raise SPYValueError(lgl, varname="show_kwargs", actual=act)

    data_x = plot_helpers.parse_toi(data, trl, show_kwargs)

    # multiple channels?
    labels = plot_helpers.parse_channel(data, show_kwargs)

    fig, ax = _plotting.mk_line_figax(ylabel='')
    _plotting.plot_lines(ax, data_x, data_y,
                         label=labels, shifted=shifted)
    fig.tight_layout()
    return fig, ax


@plot_helpers.revert_selection
def plot_SpectralData(data, **show_kwargs):

    """
    Plot either a 2d-line plot in case of
    singleton time axis or an image plot
    for time-frequency spectra.

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

    # -- how got the spectrum computed --
    method = plot_helpers.get_method(data)
    # -----------------------------------

    if method in ('wavelet', 'superlet', 'mtmconvol'):
        # multiple channels?
        label = plot_helpers.parse_channel(data, show_kwargs)
        if not isinstance(label, str):
            SPYWarning("Please select a single channel for plotting!")
            return

        # here we always need a new axes
        fig, ax = _plotting.mk_img_figax()

        # this could be more elegantly solve by
        # an in-place selection?!
        time = plot_helpers.parse_toi(data, trl, show_kwargs)
        freqs = plot_helpers.parse_foi(data, show_kwargs)

        # dimord is time x taper x freq x channel
        # need freq x time for plotting
        data_yx = data.show(**show_kwargs).T
        _plotting.plot_tfreq(ax, data_yx, time, freqs)
        ax.set_title(label, fontsize=pltConfig['sTitleSize'])
        fig.tight_layout()
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
        data_y = np.log10(data.show(**show_kwargs))

        # multiple channels?
        labels = plot_helpers.parse_channel(data, show_kwargs)

        fig, ax = _plotting.mk_line_figax(xlabel='frequency (Hz)',
                                          ylabel='power (dB)')

        _plotting.plot_lines(ax, data_x, data_y, label=labels)
        fig.tight_layout()

    return fig, ax


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
    trl = show_kwargs.get('trials', 0)
    if not isinstance(trl, int) and len(data.trials) > 1:
        SPYWarning("Please select a single trial for plotting!")
        return

    # what channel combination
    if 'channel_i' not in show_kwargs or 'channel_j' not in show_kwargs:
        SPYWarning("Please select a channel combination `channel_i` and `channel_j` for plotting!")
        return
    chi, chj = show_kwargs['channel_i'], show_kwargs['channel_j']
    # parse labels
    if isinstance(chi, str):
        chi_label = chi
    # must be int
    else:
        chi_label = f"channel{chi}"
    # parse labels
    if isinstance(chi, int):
        chi_label = f"channel{chi + 1}"
    else:
        chi_label = chi
    if isinstance(chj, int):
        chj_label = f"channel{chj + 1}"
    else:
        chj_label = chj

    # what data do we have?
    method = plot_helpers.get_method(data)
    if method == 'granger':
        xlabel = 'frequency (Hz)'
        ylabel = 'Granger causality'
        label = rf"{chi_label} $\rightarrow$ {chj_label}"
        data_x = plot_helpers.parse_foi(data, show_kwargs)
    elif method == 'coh':
        xlabel = 'frequency (Hz)'
        ylabel = 'coherence'
        label = rf"{chi_label} - {chj_label}"
        data_x = plot_helpers.parse_foi(data, show_kwargs)
    elif method == 'corr':
        xlabel = 'lag'
        ylabel = 'correlation'
        label = rf"{chi_label} - {chj_label}"
        data_x = plot_helpers.parse_toi(data, trl, show_kwargs)
    # that's all the methods we got so far
    else:
        raise NotImplementedError

    # get the data to plot
    data_y = data.show(**show_kwargs)
    if data_y.size == 0:
        lgl = "Selection with non-zero size"
        act = "got zero samples"
        raise SPYValueError(lgl, varname="show_kwargs", actual=act)

    # create the axes and figure if needed
    # persistent axes allows for plotting different
    # channel combinations into the same figure
    if not hasattr(data, 'fig') or not _plotting.ppl.fignum_exists(data.fig.number):
        data.fig, data.ax = _plotting.mk_line_figax(xlabel, ylabel)
    _plotting.plot_lines(data.ax, data_x, data_y, label=label)
    # format axes
    if method in ['granger', 'coh']:
        data.ax.set_ylim((-.02, 1.02))
    elif method == 'corr':
        data.ax.set_ylim((-1.02, 1.02))
    data.ax.legend(ncol=1)

    data.fig.tight_layout()
