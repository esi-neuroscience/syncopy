# -*- coding: utf-8 -*-
#
# Validators for user submitted frontend arguments like foi, taper, etc.
#

# Builtin/3rd party package imports
import numpy as np

from syncopy.shared.errors import SPYValueError, SPYWarning, SPYInfo
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser


def validate_foi(foi, foilim, samplerate):
        
    # Basic sanitization of frequency specifications
    if foi is not None and foilim is not None:
        lgl = "either `foi` or `foilim` specification"
        act = "both"
        raise SPYValueError(legal=lgl, varname="foi/foilim", actual=act)
    
    if foi is not None:
        if isinstance(foi, str):
            if foi == "all":
                foi = None
            else:
                raise SPYValueError(legal="'all' or `None` or list/array",
                                    varname="foi", actual=foi)
        else:
            try:
                array_parser(foi, varname="foi", hasinf=False, hasnan=False,
                             lims=[0, samplerate/2], dims=(None,))
            except Exception as exc:
                raise exc
            foi = np.array(foi, dtype="float")
            
    if foilim is not None:
        if isinstance(foilim, str):
            if foilim == "all":
                foilim = None
            else:
                raise SPYValueError(legal="'all' or `None` or `[fmin, fmax]`",
                                    varname="foilim", actual=foilim)
        else:
            try:
                array_parser(foilim, varname="foilim", hasinf=False, hasnan=False,
                             lims=[0, samplerate/2], dims=(2,))
            except Exception as exc:
                raise exc
            # foilim is of shape (2,)
            if foilim[0] > foilim[1]:
                msg = "Sorting foilim low to high.."
                SPYInfo(msg)
                foilim = np.sort(foilim)

    return foi, foilim


def validate_taper(taper,
                   tapsmofrq,
                   nTaper,
                   keeptapers,
                   foimax,
                   fs,
                   nSamples):

    '''
    General taper validation and Slepian/dpss input sanitization.
    We always want to max out nTaper to achieve the desired frequency
    smoothing bandwidth. For details about the Slepion settings see 

    "The Effective Bandwidth of a Multitaper Spectral Estimator, 
    A. T. Walden, E. J. McCoy and D. B. Percival"

    '''    

    # Warn user about DPSS only settings
    if taper != "dpss":
        if tapsmofrq is not None:
            msg = "`tapsmofrq` is only used if `taper` is `dpss`!"
            SPYWarning(msg)
        if nTaper is not None:
            msg = "`nTaper` is only used if `taper` is `dpss`!"
            SPYWarning(msg)
        if keeptapers:
            msg = "`keeptapers` is only used if `taper` is `dpss`!"
            SPYWarning(msg)

        # empty taperopt, only Slepians have options
        return {}
        
    # Set/get `tapsmofrq` if we're working w/Slepian tapers
    elif taper == "dpss":

        # minimal smoothing bandwidth in Hz
        # if sampling rate is given in Hz
        minBw = 2 * fs / nSamples
        
        # Try to derive "sane" settings by using 3/4 octave
        # smoothing of highest `foi`
        # following Hill et al. "Oscillatory Synchronization in Large-Scale
        # Cortical Networks Predicts Perception", Neuron, 2011
        # FIX ME: This "sane setting" seems quite excessive
        
        if tapsmofrq is None:
            tapsmofrq = (foimax * 2**(3 / 4 / 2) - foimax * 2**(-3 / 4 / 2)) / 2
            if tapsmofrq < minBw: # *should* not happen but just in case
                tapsmofrq = minBw
            msg = f'Automatic setting of `tapsmofrq` to {tapsmofrq:.2f}'
            SPYInfo(msg)

        # user set tapsmofrq directly
        elif tapsmofrq is not None:
            try:
                scalar_parser(tapsmofrq, varname="tapsmofrq", lims=[0, np.inf])
            except Exception as exc:
                raise exc

            if tapsmofrq < minBw:
                msg = f'Setting tapsmofrq to the minimal attainable bandwidth of {minBw:.2f}Hz'
                SPYInfo(msg)
                tapsmofrq = minBw

        # --------------------------------------------
        # set parameters for scipy.signal.windows.dpss
        NW = tapsmofrq * nSamples / (2 * fs)
        # from the minBw setting NW always is at least 1         
        Kmax = int(2 * NW - 1) # optimal number of tapers
        # --------------------------------------------
        
        # the recommended way:
        # set nTaper automatically to maximize effective smoothing bandwidth
        if nTaper is None:
            msg = f'Using {Kmax} taper(s) for multi-tapering'
            SPYInfo(msg)
            taperopt = {'NW' : NW, 'Kmax' : Kmax}
            return taperopt

        elif nTaper is not None:
            try:
                scalar_parser(nTaper,
                              varname="nTaper",
                              ntype="int_like", lims=[1, np.inf])
            except Exception as exc:
                raise exc

            if nTaper != Kmax:                
                msg = f''' 
                Manually setting the number of tapers is not recommended 
                and may (strongly) distort the spectral estimation!\n
                The optimal number of tapers is {Kmax}, you have chosen to use {nTaper}.
                '''
                SPYWarning(msg)
            
            taperopt = {'NW' : NW, 'Kmax' : nTaper}
            return taperopt
    
        
            
