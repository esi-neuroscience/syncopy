# -*- coding: utf-8 -*-
#
# Parameterization of neural power spectra with FOOOF - fitting oscillations & one over f.
#
#
#

# Builtin/3rd party package imports
import numpy as np
from fooof import FOOOF
import logging
import platform

# Constants
available_fooof_out_types = ['fooof', 'fooof_aperiodic', 'fooof_peaks']
default_fooof_opt = {'peak_width_limits': (0.5, 12.0), 'max_n_peaks': np.inf,
                     'min_peak_height': 0.0, 'peak_threshold': 2.0,
                     'aperiodic_mode': 'fixed', 'verbose': False}
available_fooof_options = list(default_fooof_opt)


def fooofspy(data_arr, in_freqs, freq_range=None,
             fooof_opt=None,
             out_type='fooof'):
    """
    Parameterization of neural power spectra using
    the FOOOF mothod by Donoghue et al: fitting oscillations & one over f.

    Parameters
    ----------
    data_arr : 2D :class:`numpy.ndarray`
         Float array containing power spectrum with shape ``(nFreq x nChannels)``,
         typically obtained from :func:`syncopy.specest.mtmfft` output. Must be in linear space. Noisy
         data will most likely lead to fitting issues, always inspect your results!
    in_freqs : 1D :class:`numpy.ndarray`
         Float array of frequencies for all spectra, typically obtained from the `freq` property
         of the `mtmfft` output (`AnalogData` object). Must not include zero.
    freq_range: float list of length 2
         optional definition of a frequency range of interest of the fooof result.
         Note: It is currently not possible for the user to set this from the frontend.
    foopf_opt : dict or None
        Additional keyword arguments passed to the `FOOOF` constructor. Available
        arguments include ``'peak_width_limits'``, ``'max_n_peaks'``, ``'min_peak_height'``,
        ``'peak_threshold'``, and ``'aperiodic_mode'``.
        Please refer to the
        `FOOOF docs <https://fooof-tools.github.io/fooof/generated/fooof.FOOOF.html#fooof.FOOOF>`_
        for the meanings and the defaults.
    out_type : string
        The requested output type, one of ``'fooof'``, ``'fooof_aperiodic'`` or ``'fooof_peaks'``.

    Returns
    -------
    Depends on the value of parameter ``'out_type'``.
    out_spectra: 2D :class:`numpy.ndarray`
        The fooofed spectrum (for out_type ``'fooof'``), the aperiodic part of the
        spectrum (for ``'fooof_aperiodic'``) or the peaks (for ``'fooof_peaks'``).
        Each row corresponds to a row in the input `data_arr`, i.e., a channel.
        The data is in linear space.
    metadata : dictionary
        Details on the model fit and settings used. Contains the following keys:
            `aperiodic_params` 2D :class:`numpy.ndarray`, the aperiodoc parameters of the fits, in log10.
            `gaussian_params` list of 2D nx3 :class:`numpy.ndarray`, the Gaussian parameters of the fits, in log10.
                              Each column describes the mean, height and width of a Gaussian fit to a peak.
            `peak_params` list of 2D xn3 :class:`numpy.ndarray`, the peak parameters (a modified version of the
                          Gaussian parameters, see FOOOF docs) of the fits, in log10. Each column describes the
                          mean, height over aperiodic and 2-sided width of a Gaussian fit to a peak.
            `n_peaks`: 1D :class:`numpy.ndarray` of int, the number of peaks detected in the spectra of the fits.
            `r_squared`: 1D :class:`numpy.ndarray` of int, the number of peaks detected in the spectra of the fits.
            `error`: 1D :class:`numpy.ndarray` of float, the model error of the fits.
            `settings_used`: dict, the settings used, including the keys `fooof_opt`, `out_type`, and `freq_range`.

    Examples
    --------
    Run fooof on a generated power spectrum:
    >>> from syncopy.specest.fooofspy import fooofspy
    >>> from fooof.sim.gen import gen_power_spectrum
    >>> freqs, powers = gen_power_spectrum([3, 40], [1, 1], [[10, 0.2, 1.25], [30, 0.15, 2]])
    >>> spectra, metadata = fooofspy(powers, freqs, out_type='fooof')

    References
    -----
    Donoghue T, Haller M, Peterson EJ, Varma P, Sebastian P, Gao R, Noto T, Lara AH, Wallis JD,
    Knight RT, Shestyuk A, & Voytek B (2020). Parameterizing neural power spectra into periodic
    and aperiodic components. Nature Neuroscience, 23, 1655-1665.
    DOI: 10.1038/s41593-020-00744-x
    """

    if data_arr.ndim < 2:  # Attach dummy channel axis for single channel data.
        data_arr = data_arr[:, np.newaxis]

    if fooof_opt is None:
        fooof_opt = default_fooof_opt
    else:
        fooof_opt = {**default_fooof_opt, **fooof_opt}

    if in_freqs is None:
        raise ValueError('infreqs: The input frequencies are required and must not be None.')

    logger = logging.getLogger("syncopy_" + platform.node())
    logger.debug(f"Running FOOOF backend function on data chunk with shape {data_arr.shape}.")

    invalid_fooof_opts = [i for i in fooof_opt.keys() if i not in available_fooof_options]
    if invalid_fooof_opts:
        raise ValueError("fooof_opt: invalid keys: '{inv}', allowed keys are: '{lgl}'.".format(inv=invalid_fooof_opts, lgl=fooof_opt.keys()))

    if out_type not in available_fooof_out_types:
        raise ValueError("out_type: invalid value '{inv}', expected one of '{lgl}'.".format(inv=out_type, lgl=available_fooof_out_types))

    if in_freqs.size != data_arr.shape[0]:
        raise ValueError("data_arr/in_freqs: The signal length {sl} must match the number of frequency labels {ll}.".format(sl=data_arr.shape[0], ll=in_freqs.size))

    if in_freqs[0] == 0:
        raise ValueError("in_freqs: invalid frequency range {minf} to {maxf}, expected a frequency range that does not include zero.".format(minf=min(in_freqs), maxf=max(in_freqs)))

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
    gaussian_params = list()  # Gaussian fit parameters of peaks
    peak_params = list()  # Peak fit parameters, a modified version of gaussian_parameters. See FOOOF docs.

    # Run fooof and store results.
    for channel_idx in range(num_channels):
        spectrum = data_arr[:, channel_idx]
        fm.fit(in_freqs, spectrum, freq_range=freq_range)

        # compute aperiodic fit
        offset = fm.aperiodic_params_[0]
        if fm.aperiodic_mode == 'fixed':
            exp = fm.aperiodic_params_[1]
            aperiodic_spec = offset - np.log10(in_freqs**exp)
        else:  # fm.aperiodic_mode == 'knee':
            knee = fm.aperiodic_params_[1]
            exp = fm.aperiodic_params_[2]
            aperiodic_spec = offset - np.log10(knee + in_freqs**exp)

        if out_type == 'fooof':
            out_spectrum = 10 ** fm.fooofed_spectrum_  # The powers. Need to undo log10, which is used internally by fooof.
        elif out_type == "fooof_aperiodic":
            out_spectrum = 10 ** aperiodic_spec
        elif out_type == "fooof_peaks":
            out_spectrum = (10 ** fm.fooofed_spectrum_) - (10 ** aperiodic_spec)
            out_spectrum += 1e-16  # Prevent zero values in areas without peaks/periodic parts. These would result in log plotting issues.
        else:
            raise ValueError("out_type: invalid value '{inv}', expected one of '{lgl}'.".format(inv=out_type, lgl=available_fooof_out_types))

        out_spectra[:, channel_idx] = out_spectrum
        aperiodic_params[:, channel_idx] = fm.aperiodic_params_
        n_peaks[channel_idx] = fm.n_peaks_
        r_squared[channel_idx] = fm.r_squared_
        error[channel_idx] = fm.error_
        gaussian_params.append(fm.gaussian_params_)
        peak_params.append(fm.peak_params_)

    settings_used = {'fooof_opt': fooof_opt, 'out_type': out_type, 'freq_range': freq_range}
    #  Note: we add the 'settings_used' here in the backend, but they get stripped in the middle layer
    #       (in the 'compRoutines.py/fooofspy_cF()'), so they do not reach the frontend.
    #        The reason for removing them there is that we/h5py do not support nested dicts as
    #        dataset/group attributes, and thus we cannot encode them in hdf5. We could work around
    #        that, but due to our log, we do not really need to.
    #        Returning them from here still has the benefit that we can test for them in backend tests.
    metadata = {'aperiodic_params': aperiodic_params, 'gaussian_params': gaussian_params,
               'peak_params': peak_params, 'n_peaks': n_peaks, 'r_squared': r_squared,
               'error': error, 'settings_used': settings_used}

    return out_spectra, metadata

