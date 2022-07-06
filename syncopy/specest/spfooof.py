# -*- coding: utf-8 -*-
#
# Parameterization of neural power spectra with FOOOF - fitting oscillations & one over f.
#
#
#

# Builtin/3rd party package imports
import numpy as np
from fooof import FOOOF

# Syncopy imports
from syncopy.shared.errors import SPYValueError
from syncopy.shared.const_def import fooofDTypes

# Constants
available_fooof_out_types = fooofDTypes.keys()
available_fooof_options = ['peak_width_limits', 'max_n_peaks', 
                           'min_peak_height', 'peak_threshold',
                           'aperiodic_mode', 'verbose']


def spfooof(data_arr, in_freqs, freq_range=None,
            fooof_opt={'peak_width_limits': (0.5, 12.0), 'max_n_peaks': np.inf,
                       'min_peak_height': 0.0, 'peak_threshold': 2.0,
                       'aperiodic_mode': 'fixed', 'verbose': True},
            out_type='fooof'):
    """
    Parameterization of neural power spectra using 
    the FOOOF mothod by Donoghue et al: fitting oscillations & one over f.

    Parameters
    ----------
    data_arr : 2D :class:`numpy.ndarray`
         Float array containing power spectrum with shape ``(nFreq x nChannels)``,
         typically obtained from :func:`syncopy.specest.mtmfft` output.
    in_freqs : 1D :class:`numpy.ndarray`
         Float array of frequencies for all spectra, typically obtained from mtmfft output.
    freq_range: 2-tuple
         optional definition of a frequency range of interest of the fooof result (post processing).
    foof_opt : dict or None
        Additional keyword arguments passed to the `FOOOF` constructor. Available
        arguments include 'peak_width_limits', 'max_n_peaks', 'min_peak_height',
        'peak_threshold', and 'aperiodic_mode'.
        Please refer to the
        `FOOOF docs <https://fooof-tools.github.io/fooof/generated/fooof.FOOOF.html#fooof.FOOOF>`_
        for the meanings.
    out_type : string
        The requested output type, one of ``'fooof'``, 'fooof_aperiodic' or 'fooof_peaks'.

    Returns
    -------
    Depends on the value of parameter ``'out_type'``.
    out_spectra: 2D :class:`numpy.ndarray`
        The fooofed spectrum (for out_type ``'fooof'``), the aperiodic part of the
        spectrum (for ``'fooof_aperiodic'``) or the peaks (for ``'fooof_peaks'``).
    details : dictionary
        Details on the model fit and settings used.

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
        fooof_opt = {'peak_width_limits': (0.5, 12.0), 'max_n_peaks': np.inf,
                     'min_peak_height': 0.0, 'peak_threshold': 2.0,
                     'aperiodic_mode': 'fixed', 'verbose': True}

    if out_type not in available_fooof_out_types:
        lgl = "'" + "or '".join(opt + "' " for opt in available_fooof_out_types)
        raise SPYValueError(legal=lgl, varname="out_type", actual=out_type)

    if in_freqs is None:
        raise SPYValueError(legal='The input frequencies are required and must not be None.', varname='in_freqs')
    print("number of fooof input freq labels: %d" % (in_freqs.size))

    if in_freqs.size != data_arr.shape[0]:
        raise SPYValueError(legal='The signal length %d must match the number of frequency labels %d.' % (data_arr.shape[0], in_freqs.size), varname="data_arr/in_freqs")

    num_channels = data_arr.shape[1]

    fm = FOOOF(**fooof_opt)

    # Prepare output data structures
    out_spectra = np.zeros_like(data_arr, data_arr.dtype)
    if fm.aperiodic_mode == 'knee':
        aperiodic_params = np.zeros(shape=(3, num_channels), dtype=np.float64)
    else:
        aperiodic_params = np.zeros(shape=(2, num_channels), dtype=np.float64)
    n_peaks = np.zeros(shape=(num_channels), dtype=np.int32)    # helper: number of peaks fit.
    r_squared = np.zeros(shape=(num_channels), dtype=np.float64)  # helper: R squared of fit.
    error = np.zeros(shape=(num_channels), dtype=np.float64)      # helper: model error.

    # Run fooof and store results. We could also use a fooof group.
    for channel_idx in range(num_channels):
        spectrum = data_arr[:, channel_idx]
        fm.fit(in_freqs, spectrum, freq_range=freq_range)

        if out_type == 'fooof':
            out_spectrum = fm.fooofed_spectrum_  # the powers
        elif out_type == "fooof_aperiodic":
            offset = fm.aperiodic_params_[0]
            if fm.aperiodic_mode == 'fixed':
                exp = fm.aperiodic_params_[1]
                out_spectrum = offset - np.log10(in_freqs**exp)
            else:  # fm.aperiodic_mode == 'knee':
                knee = fm.aperiodic_params_[1]
                exp = fm.aperiodic_params_[2]
                out_spectrum = offset - np.log10(knee + in_freqs**exp)
        elif out_type == "fooof_peaks":
            gp = fm.gaussian_params_
            out_spectrum = np.zeros_like(in_freqs, in_freqs.dtype)
            for row_idx in range(len(gp)):
                ctr, hgt, wid = gp[row_idx, :]
                # Extract Gaussian parameters: central frequency (=mean), power over aperiodic, bandwith of peak (= 2* stddev of Gaussian).
                # see FOOOF docs for details, especially Tutorial 2, Section 'Notes on Interpreting Peak Parameters'
                out_spectrum = out_spectrum + hgt * np.exp(- (in_freqs - ctr)**2 / (2 * wid**2))
        else:
            raise SPYValueError(legal=available_fooof_out_types, varname="out_type", actual=out_type)

        print("Channel %d fooofing done, received spektrum of length %d." % (channel_idx, out_spectrum.size))

        out_spectra[:, channel_idx] = out_spectrum
        aperiodic_params[:, channel_idx] = fm.aperiodic_params_
        n_peaks[channel_idx] = fm.n_peaks_
        r_squared[channel_idx] = fm.r_squared_
        error[channel_idx] = fm.error_

    settings_used = {'fooof_opt': fooof_opt, 'out_type': out_type, 'freq_range': freq_range}
    details = {'aperiodic_params': aperiodic_params, 'n_peaks': n_peaks, 'r_squared': r_squared, 'error': error, 'settings_used': settings_used}

    return out_spectra, details
