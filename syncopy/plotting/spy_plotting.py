# -*- coding: utf-8 -*-
# 
# Syncopy plotting routines
# 
# Created: 2020-03-17 17:33:35
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-04-02 11:57:01>

# # Builtin/3rd party package imports
# import tensorlfow

# Local imports
from syncopy.shared.kwarg_decorators import unwrap_cfg
from syncopy import __plt__

# Conditional imports
if __plt__:
    import matplotlib.pyplot as plt 
    # import matplotlib.style as mplstyle
    # import matplotlib as mpl
    # plt.ion()
    # mplstyle.use("fast")
    # mpl.rcParams['lines.linewidth'] = 2
    # mpl.rcParams['lines.linestyle'] = '--'    

__all__ = ["singleplot", "multiplot"]
msg = "Plotting not available"


@unwrap_cfg
def singleplot(data, trials=None, channels=None, toi=None, toilim=None, foi=None,
               foilim=None, tapers=None, units=None, eventids=None, overlay=False, 
               out=None, **kwargs):
    """
    Coming soon...
    
    FT_SINGLEPLOTER plots the event-related fields or potentials of a single
    channel or the average over multiple channels. Multiple datasets can be
    overlayed.

    Use as
    ft_singleplotER(cfg, data)
    or
    ft_singleplotER(cfg, data1, data2, ..., datan)    
    
    overlay: monkey-patch axis object for trial-list!
    
    unwrap_cfg: make the decorator compatible w/list-like data inputs so that it 
    can handle things like `singleplot(data1, data2, data3, **kwargs)`
    
    overlay: monkey-patch axis object for trial-list!
    """
    
    if not __plt__:
        print('ASDF')
    pass


@unwrap_cfg
def multiplot(data, trials=None, channels=None, toi=None, toilim=None, foi=None,
               foilim=None, tapers=None, units=None, eventids=None, 
               out=None, **kwargs):
    """
    Coming soon...
    
    For AnalogData: default dimord works w/matplotlib, i.e., 
    
    plot(data) with data nSample x nChannel generates nChannel 2DLines 
    """
    pass
