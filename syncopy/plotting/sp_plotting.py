# -*- coding: utf-8 -*-
#
# The singlepanel plotting functions for Syncopy
# data types
# 1st argument **must** be `spy_data`
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
def plot_AnalogData(spy_data, shifted=True, **show_kwargs):
    """
    Simple 2d-line plot of selected channels.

    Parameters
    ----------
    spy_data : :class:`~syncopy.datatype.AnalogData`
    shifted : bool
        Stacks the signals on top of each other if `True` by
        extending the y-axis
    show_kwargs : :func:`~syncopy.datatype.methods.show.show` arguments

    Returns
    -------
    fig : `matplotlib.figure.Figure` instance (or `None` in case of errors), the plot figure.
    ax  : `matplotlib.axes.Axes` instance (or `None` in case of errors), the plot axes.
    """

    if not __plt__:
        SPYWarning(pltErrMsg)
        return None, None

    # right now we have to enforce
    # single trial selection only
    trl = show_kwargs.get("trials", None)
    if not isinstance(trl, Number) and len(spy_data.trials) > 1:
        SPYWarning("Please select a single trial for plotting.")
        return None, None
    # only 1 trial so no explicit selection needed
    elif len(spy_data.trials) == 1:
        trl = 0

    # get the data to plot
    data_y = spy_data.show(**show_kwargs)
    # 'time' and 'channel' are the only axes
    if spy_data._defaultDimord != spy_data.dimord:
        data_y = data_y.T
    if data_y.size == 0:
        lgl = "Selection with non-zero size"
        act = "got zero samples"
        raise SPYValueError(lgl, varname="show_kwargs", actual=act)

    data_x = plot_helpers.parse_toi(spy_data, trl, show_kwargs)

    # multiple channels?
    labels = plot_helpers.parse_channel(spy_data, show_kwargs)

    fig, ax = _plotting.mk_line_figax(ylabel="")
    _plotting.plot_lines(ax, data_x, data_y, label=labels, shifted=shifted)
    fig.tight_layout()
    return fig, ax


@plot_helpers.revert_selection
def plot_SpectralData(spy_data, logscale=True, **show_kwargs):
    """
    Plot either a 2d-line plot in case of
    singleton time axis or an image plot
    for time-frequency spectra.

    Parameters
    ----------
    spy_data : :class:`~syncopy.datatype.SpectralData`
    logscale : bool
        If `True` the log10 of the power spectra (output='pow') values
        is plotted.
    show_kwargs : :func:`~syncopy.datatype.methods.show.show` arguments

    Returns
    -------
    fig : `matplotlib.figure.Figure` instance (or `None` in case of errors), the plot figure.
    ax  : `matplotlib.axes.Axes` instance (or `None` in case of errors), the plot axes.
    """

    if not __plt__:
        SPYWarning(pltErrMsg)
        return None, None

    # right now we have to enforce
    # single trial selection only
    trl = show_kwargs.get("trials", None)
    if not isinstance(trl, Number) and len(spy_data.trials) > 1:
        SPYWarning("Please select a single trial for plotting.")
        return None, None
    elif len(spy_data.trials) == 1:
        trl = 0

    is_tf = plot_helpers.check_if_time_freq(spy_data)

    if is_tf:
        # multiple channels?
        label = plot_helpers.parse_channel(spy_data, show_kwargs)
        # only relevant for mtmconvol
        if "taper" in show_kwargs:
            SPYWarning("Taper selection not supported for time-frequency spectra!\nSkipping plot..")
            return None, None

        if not isinstance(label, str):
            SPYWarning("Please select a single channel for plotting!\nSkipping plot..")
            return None, None

        # here we always need a new axes
        fig, ax = _plotting.mk_img_figax()

        time = plot_helpers.parse_toi(spy_data, trl, show_kwargs)
        freqs = plot_helpers.parse_foi(spy_data, show_kwargs)

        # custom dimords for SpectralData not supported atm
        # dimord is time x taper x freq x channel
        # need freq x time for plotting
        data_yx = spy_data.show(**show_kwargs).T
        _plotting.plot_tfreq(ax, data_yx, time, freqs)
        ax.set_title(label, fontsize=pltConfig["sTitleSize"])
        fig.tight_layout()

    # just a line plot
    else:

        msg = False
        if "latency" in show_kwargs:
            show_kwargs.pop("latency")
            msg = True
        if msg:
            msg = "Line spectra don't have a time axis, " "ignoring `toi/toilim` selection!"
            SPYWarning(msg)

        # multiple channels?
        channels = plot_helpers.parse_channel(spy_data, show_kwargs)

        # just multiple tapers or multiple channels in one plot
        if len(spy_data.taper) != 1:
            taper = show_kwargs.get("taper")
            if not isinstance(taper, (Number, str)) and not isinstance(channels, str):
                msg = "Please select a single taper or a single channel \nfor plotting multi-taper spectra.. aborting plotting\n"
                SPYWarning(msg)
                return None, None
            # single channel, multiple tapers
            elif isinstance(channels, str):
                labels = spy_data.taper
            # single taper, multiple channels
            elif isinstance(taper, (Number, str)):
                labels = channels
        else:
            labels = channels
        # get the data to plot
        data_x = plot_helpers.parse_foi(spy_data, show_kwargs)
        output = plot_helpers.get_output(spy_data, "freqanalysis")

        pow_or_fooof = "fooof" in output or output == "pow"

        # only log10 the absolute squared spectra
        if pow_or_fooof and logscale:
            data_y = np.log10(spy_data.show(**show_kwargs))
            ylabel = "power (dB)"
        elif output in ["fourier", "complex"]:
            SPYWarning(
                "Can't plot complex valued spectra, choose 'real' or 'imag' as freqanalysis output.. aborting plotting"
            )
            return None, None
        else:
            data_y = spy_data.show(**show_kwargs)
            ylabel = f"{output} (a.u.)"

        # for itc.. needs to be improved
        if output is None:
            ylabel = ""

        # flip if required
        if data_y.ndim > 1:
            if data_y.shape[1] == len(data_x):
                data_y = data_y.T

        fig, ax = _plotting.mk_line_figax(xlabel="frequency (Hz)", ylabel=ylabel)

        _plotting.plot_lines(ax, data_x, data_y, label=labels, lw=1.5, alpha=0.8)
        fig.tight_layout()

    return fig, ax


@plot_helpers.revert_selection
def plot_CrossSpectralData(spy_data, **show_kwargs):
    """
    Plot 2d-line plots for the different connectivity measures.

    Parameters
    ----------
    spy_data : :class:`~syncopy.datatype.CrossSpectralData`
    show_kwargs : :func:`~syncopy.datatype.methods.show.show` arguments

    Returns
    -------
    fig : `matplotlib.figure.Figure` instance (or `None` in case of errors), the plot figure.
    ax  : `matplotlib.axes.Axes` instance (or `None` in case of errors), the plot axes.
    """

    if not __plt__:
        SPYWarning(pltErrMsg)
        return None, None

    # right now we have to enforce
    # single trial selection only
    trl = show_kwargs.get("trials", 0)
    if not isinstance(trl, int) and len(spy_data.trials) > 1:
        SPYWarning("Please select a single trial for plotting.")
        return None, None

    # what channel combination
    if "channel_i" not in show_kwargs or "channel_j" not in show_kwargs:
        SPYWarning("Please select a channel combination `channel_i` and `channel_j` for plotting.")
        return None, None
    chi, chj = show_kwargs["channel_i"], show_kwargs["channel_j"]
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
    method = plot_helpers.get_method(spy_data, "connectivityanalysis")
    output = plot_helpers.get_output(spy_data, "connectivityanalysis")

    if method == "granger":
        xlabel = "frequency (Hz)"
        ylabel = "Granger causality"
        label = rf"{chi_label} $\rightarrow$ {chj_label}"
        data_x = plot_helpers.parse_foi(spy_data, show_kwargs)
    elif method == "coh":
        xlabel = "frequency (Hz)"
        ylabel = f"{output} coherence"
        label = rf"{chi_label} - {chj_label}"
        data_x = plot_helpers.parse_foi(spy_data, show_kwargs)
    elif method == "ppc":
        xlabel = "frequency (Hz)"
        ylabel = "PPC"
        label = rf"{chi_label} - {chj_label}"
        data_x = plot_helpers.parse_foi(spy_data, show_kwargs)
    elif method == "corr":
        xlabel = "lag"
        ylabel = "correlation"
        label = rf"{chi_label} - {chj_label}"
        data_x = plot_helpers.parse_toi(spy_data, trl, show_kwargs)
    # that's all the methods we got so far
    else:
        raise NotImplementedError

    is_tf = plot_helpers.check_if_time_freq(spy_data)

    # time dependent coherence
    if method in ["coh", "ppc"] and is_tf:
        # here we always need a new axes
        fig, ax = _plotting.mk_img_figax()

        time = plot_helpers.parse_toi(spy_data, trl, show_kwargs)
        freqs = plot_helpers.parse_foi(spy_data, show_kwargs)

        # custom dimords for SpectralData not supported atm
        # dimord is time x freq x channel_i x channel_j
        # need freq x time for plotting
        data_yx = spy_data.show(**show_kwargs).T
        _plotting.plot_tfreq(ax, data_yx, time, freqs, cmap="cividis")
        ax.set_title(f"{method}: " + label, fontsize=pltConfig["sTitleSize"])
        fig.tight_layout()

        return fig, ax

    else:
        # get the data to plot
        data_y = spy_data.show(**show_kwargs)
        if data_y.size == 0:
            lgl = "Selection with non-zero size"
            act = f"{show_kwargs}, got zero samples"
            raise SPYValueError(lgl, varname="show_kwargs", actual=act)

        # create the axes and figure if needed
        # persistent axes allows for plotting different
        # channel combinations into the same figure
        if not hasattr(spy_data, "fig") or not _plotting.ppl.fignum_exists(spy_data.fig.number):
            spy_data.fig, spy_data.ax = _plotting.mk_line_figax(xlabel, ylabel)
        _plotting.plot_lines(spy_data.ax, data_x, data_y, label=label)
        # format axes
        if method in ["granger", "coh"] and output in ["pow", "abs"]:
            spy_data.ax.set_ylim((-0.02, 1.02))
        elif method == "corr":
            spy_data.ax.set_ylim((-1.02, 1.02))
        spy_data.ax.legend(ncol=1)

        spy_data.fig.tight_layout()

        return spy_data.fig, spy_data.ax
