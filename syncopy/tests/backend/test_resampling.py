# -*- coding: utf-8 -*-
#
# syncopy.preproc resampling tests
#
import numpy as np
import matplotlib.pyplot as ppl

from syncopy.preproc import resampling, firws
from syncopy.specest import mtmfft


def test_resample():

    """
    Tests both the non-trivial resampling method
    with non-integer sampling rate division (polyphase method)
    and also the much simpler downsampling (leave out every
    nth sample)

    Test data is comprised of white noise, criteria
    are the (absence of) gain in the remaining frequency
    band after resampling. 

    This strategy was inspired by a Field Trip Tutorial written
    by Jan Mathijs: `https://www.fieldtriptoolbox.org/faq/resampling_lowpassfilter/`_
    """

    nSamples = 1000
    nTrials = 100
    orig_fs = 500 #Hz
    data = [np.random.randn(nSamples) for _ in range(nTrials)]
    
    # get original trial-averaged power spectrum
    orig_power, orig_freqs = trl_av_power(data, nSamples, orig_fs)
    
    # -- test simple downsampling w/o low-pass filtering --
    
    ds_fs = orig_fs // 2 # half the original sampling rate
    # make sure it's an integer division
    assert orig_fs % ds_fs == 0    
    ds_data = [resampling.downsample(signal, orig_fs, ds_fs) for signal in data]
    ds_power, ds_freqs = trl_av_power(ds_data, nSamples, ds_fs)

    # w/o low-pass filtering, the high frequencies above the
    # new Nyquist frequency wrap around and give a gain
    # directly proportional to the ratio orig_fs / ds_fs
    gain = ds_power.mean() / orig_power.mean()
    fs_ratio = orig_fs / ds_fs
    assert 0.98 * fs_ratio < gain < 1.02 * fs_ratio

    # -- test simple downsampling with low-pass filtering --
    
    # design filter with cut off at new Nyquist
    lpfilter = firws.design_wsinc("blackman", order = nSamples, f_c = 0.5 / fs_ratio)
    # apply to all signals BEFORE downsampling
    lp_data = [firws.apply_fir(signal, lpfilter) for signal in data]
    ds_lp_data = [resampling.downsample(signal, orig_fs, ds_fs) for signal in lp_data]
    ds_lp_power, ds_freqs = trl_av_power(ds_lp_data, nSamples, ds_fs)

    # with low-pass filtering, the high frequencies above the
    # new Nyquist frequency are removed and hence there should
    # be no gain
    gain = ds_lp_power.mean() / orig_power.mean()
    assert 0.98  < gain < 1.02

    # -- test non-trivial resampling --

    rs_fs = 200
    # make sure we have a non-integer division
    assert orig_fs % rs_fs > 1 # strictly > 0 would be enough..
    
    rs_data = [resampling.resample(signal, orig_fs, ds_fs) for signal in data]
    rs_power, rs_freqs = trl_av_power(data, nSamples, ds_fs)
    
    # here we have implicit FIR filtering built in,
    # hence there should be again no gain    
    gain = rs_power.mean() / orig_power.mean()
    assert 0.98  < gain < 1.02
    
    # -- plot all the power spectra --
    
    fig, ax = ppl.subplots()
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('power (a.u.)')

    ax.plot(orig_freqs, orig_power, label='original')
    ax.plot(ds_freqs, ds_power, label='downsampled')
    ax.plot(ds_freqs, ds_lp_power, label='downsampled + FIR')
    ax.plot(rs_freqs, rs_power, label='resampled')

    ax.legend()
    
def trl_av_power(data, nSamples, fs, tapsmofrq=1):

    power = []
    for signal in data:
        NW, Kmax = mtmfft._get_dpss_pars(1, nSamples, fs)
        ftr, freqs = mtmfft.mtmfft(signal, samplerate=fs,
                                        taper='dpss',
                                        taper_opt={'Kmax' : Kmax, 'NW' : NW})
        # no tapers so 1st axis is singleton
        power.append(np.real(ftr * ftr.conj()).mean(axis=0))
    power = np.mean(power, axis=0)
    return power, freqs

