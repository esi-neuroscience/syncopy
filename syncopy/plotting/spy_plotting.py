# -*- coding: utf-8 -*-
# 
# Syncopy plotting routines
# 
# Created: 2020-03-17 17:33:35
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-05-06 14:29:30>

# # Builtin/3rd party package imports
# import tensorlfow

# Local imports
from syncopy.shared.kwarg_decorators import unwrap_cfg
from syncopy.shared.errors import SPYError, SPYTypeError, SPYWarning
from syncopy.shared.parsers import data_parser
from syncopy.shared.tools import get_defaults
from syncopy import __plt__
import syncopy as spy  # FIXME: for WIP type-error-checking

# Conditional imports
if __plt__:
    import matplotlib.pyplot as plt 
    import matplotlib.style as mplstyle
    import matplotlib as mpl
    plt.ion()
    mplstyle.use("fast")
    mpl.rcParams["figure.dpi"] = 100
    
# Global style settings for single-/multi-plots
pltConfig = {"singleTitleSize": 12,
             "singleLabelSize": 10,
             "singleTickSize": 8,
             "singleLegendSize": 10,
             "singleFigSize": (6.4, 4.8),
             "multiTitleSize": 10,
             "multiLabelSize": 8,
             "multiTickSize": 6,
             "multiLegendSize": 8,
             "multiFigSize": (10, 6.8)}

# Global consistent error message if matplotlib is missing
pltErrMsg = "Could not import 'matplotlib': {} requires a working matplotlib installation!"

__all__ = ["singleplot", "multiplot"]


@unwrap_cfg
def singleplot(*data, 
               trials="all", channels="all", toilim=None, 
               avg_channels=True, 
               title=None, grid=None, overlay=True, fig=None, **kwargs):
    """
    Coming soon...
    
    overlay : True -> overlay figure if more than one data object is provided
    
    =====================
    
    FT_SINGLEPLOTER plots the event-related fields or potentials of a single
    channel or the average over multiple channels. Multiple datasets can be
    overlayed.

    Use as
    ft_singleplotER(cfg, data)
    or
    ft_singleplotER(cfg, data1, data2, ..., datan)    
    
    unwrap_cfg: make the decorator compatible w/list-like data inputs so that it 
    can handle things like `singleplot(data1, data2, data3, **kwargs)`
    
    If you specify multiple datasets they should contain the same channels, etc.
    """
    
    # Abort if matplotlib is not available
    if not __plt__:
        raise SPYError(pltErrMsg.format("singleplot"))
    
    # Collect all keywords of corresponding class-method (w/possibly user-provided 
    # values) in dictionary 
    defaults = get_defaults(data[0].singleplot)
    lcls = locals()
    kwords = {}
    for kword in defaults:
        kwords[kword] = lcls[kword]

    # Call plotting manager
    return _anyplot(*data, overlay=overlay, method="singleplot", **kwords, **kwargs)


@unwrap_cfg
def multiplot(*data, 
              trials="all", channels="all", toilim=None, 
              avg_channels=True, avg_trials=True, 
              title=None, grid=None, overlay=True, fig=None, **kwargs):
    """
    Coming soon...
    
    For AnalogData: default dimord works w/matplotlib, i.e., 
    
    plot(data) with data nSample x nChannel generates nChannel 2DLines 
    """

    # Abort if matplotlib is not available
    if not __plt__:
        raise SPYError(pltErrMsg.format("multiplot"))

    # Collect all keywords of corresponding class-method (w/possibly user-provided 
    # values) in dictionary 
    defaults = get_defaults(data[0].multiplot)
    lcls = locals()
    kwords = {}
    for kword in defaults:
        kwords[kword] = lcls[kword]

    # Call plotting manager
    return _anyplot(*data, overlay=overlay, method="multiplot", **kwords, **kwargs)


def _anyplot(*data, overlay=None, method=None, **kwargs):
    """
    Coming soon...
    """

    # The only error-checking done in here: ensure `overlay` is Boolean and assert 
    # `data` contains only non-empty Syncopy objects
    if not isinstance(overlay, bool):
        raise SPYTypeError(overlay, varname="overlay", expected="bool")
    for obj in data:
        try:
            data_parser(obj, varname="data", empty=False)
        except Exception as exc:
            raise exc
        # FIXME: while plotting is still WIP
        if not isinstance(obj, spy.AnalogData):
            errmsg = "Plotting currently only supported for `AnalogData` objects"
            raise NotImplementedError(errmsg)

    # See if figure was provided
    start = 0
    nData = len(data)
    fig = kwargs.pop("fig", None)
    if not overlay and fig is not None and nData > 1:
        msg = "User-provided figures not supported for non-overlay visualization " +\
            "of {} datasets. Supplied figure will not be used. "
        SPYWarning(msg.format(nData))
        fig = None

    # If we're overlaying, preserve initial figure object to plot over iteratively        
    if overlay:
        fig = getattr(data[0], method)(fig=fig, **kwargs)
        start = 1
    figList = []
    for n in range(start, nData):
        figList.append(getattr(data[n], method)(fig=fig, **kwargs))

    # Return single figure object (if `overlay` is `True`) or list of mulitple figs    
    if overlay:
        return fig
    return figList
