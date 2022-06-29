# -*- coding: utf-8 -*-
#
# Parameterization of neural power spectra with FOOOF - fitting oscillations & one over f.
#
#
#

# Builtin/3rd party package imports
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit

# Syncopy imports
from syncopy.shared.errors import SPYValueError

# Constants
available_fooof_out_types = ['spec_periodic', 'fit_gaussians', 'fit_aperiodic']
available_fooof_options = ['peak_width_limits', 'max_n_peaks', 'min_peak_height', 'peak_threshold', 'aperiodic_mode', 'verbose']

def fooof(data_arr,
           freqs,
           fooof_opt= {'peak_width_limits' : (0.5, 12.0), 'max_n_peaks':np.inf, 'min_peak_height':0.0, 'peak_threshold':2.0, 'aperiodic_mode':'fixed', 'verbose':True},
           out_type='fooof'):
    """
    Parameterization of neural power spectra using 
    the FOOOF mothod by Donoghue et al: fitting oscillations & one over f.

    Parameters
    ----------
    data_arr : 3D :class:`numpy.ndarray`
         Complex has shape ``(nTapers x nFreq x nChannels)``, obtained from :func:`syncopy.specest.mtmfft` output.
    freqs : 1D :class:`numpy.ndarray`
         Array of Fourier frequencies, obtained from mtmfft output.
    foof_opt : dict or None
        Additional keyword arguments passed to the `FOOOF` constructor. Available
        arguments include 'peak_width_limits', 'max_n_peaks', 'min_peak_height',
        'peak_threshold', and 'aperiodic_mode'.
        Please refer to the
        `FOOOF docs <https://fooof-tools.github.io/fooof/generated/fooof.FOOOF.html#fooof.FOOOF>`_
        for the meanings.
    out_type : string
        The requested output type, one of ``'spec_periodic'`` for the original spectrum minus the aperiodic
        parts, ``'fit_gaussians'`` for the Gaussians fit to the original spectrum minus the aperiodic parts, or
        ``'fit_aperiodic'``.

    Returns
    -------
    Depends on the value of parameter ``'out_type'``.
    TODO: describe here.

    References
    -----
    Donoghue T, Haller M, Peterson EJ, Varma P, Sebastian P, Gao R, Noto T, Lara AH, Wallis JD,
    Knight RT, Shestyuk A, & Voytek B (2020). Parameterizing neural power spectra into periodic
    and aperiodic components. Nature Neuroscience, 23, 1655-1665.
    DOI: 10.1038/s41593-020-00744-x
    """

    # attach dummy channel axis in case only a
    # single signal/channel is the input
    if data_arr.ndim < 2:
        data_arr = data_arr[:, np.newaxis]

    if fooof_opt is None:
        fooof_opt = {'peak_width_limits' : (0.5, 12.0), 'max_n_peaks':np.inf, 'min_peak_height':0.0, 'peak_threshold':2.0, 'aperiodic_mode':'fixed', 'verbose':True}

    if out_type not in available_fooof_out_types:
        lgl = "'" + "or '".join(opt + "' " for opt in available_fooof_out_types)
        raise SPYValueError(legal=lgl, varname="out_type", actual=out_type)

    
    
    return data_arr

