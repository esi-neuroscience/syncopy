import numpy as np
import matplotlib.pyplot as ppl
from scipy.signal import windows

from syncopy.specest import mtmfft
from syncopy.specest import mtmconvol
from syncopy.specest import superlet, wavelet
from syncopy.specest import wavelets as spywave


def gen_testdata(freqs=[20, 40, 60],
                 cycles=11, fs=1000,
                 eps = 0):

    """
    Harmonic superposition of multiple
    few-cycle oscillations akin to the
    example of Figure 3 in Moca et al. 2021 NatComm

    Each harmonic has a frequency neighbor with +10Hz
    and a time neighbor after 2 cycles(periods).
    """

    signal = []
    for freq in freqs:

        # 10 cycles of f1
        tvec = np.arange(cycles / freq, step=1 / fs)

        harmonic = np.cos(2 * np.pi * freq * tvec)
        # frequency neighbor
        f_neighbor = np.cos(2 * np.pi * (freq + 10) * tvec)
        packet = harmonic +  f_neighbor

        # 2 cycles time neighbor
        delta_t = np.zeros(int(2 / freq * fs))

        # 5 cycles break
        pad = np.zeros(int(5 / freq * fs))

        signal.extend([pad, packet, delta_t, harmonic])

    # stack the packets together with some padding
    signal.append(pad)
    signal = np.concatenate(signal)

    # additive white noise
    if eps > 0:
        signal = np.random.randn(len(signal)) * eps + signal

    return signal


fs = 1000 # sampling frequency

# generate 3 packets at 20, 40 and 60Hz with 10 cycles each
# Noise variance is given by eps
signal_freqs = np.array([20, 50, 80])

# signal_freqs = np.array([20, 70])
cycles = 12
A = 5 # signal amplitude
signal = A * gen_testdata(freqs=signal_freqs, cycles=cycles, fs=fs, eps=0.)

# define frequencies of interest for wavelet methods
foi = np.arange(1, 101, step=1)

# closest spectral indices to validate time-freq results
freq_idx = []
for frequency in signal_freqs:
    freq_idx.append(np.argmax(foi >= frequency))


def test_mtmconvol():

    # 10 cycles of 40Hz are 250 samples
    window_size = 750

    # default - stft pads with 0's to make windows fit
    # we choose N-1 overlap to retrieve a time-freq estimate
    # for each epoch in the signal

    # the transforms have shape (nTime, nTaper, nFreq, nChannel)
    ftr, freqs = mtmconvol.mtmconvol(signal,
                                     samplerate=fs, taper='cosine',
                                     nperseg=window_size,
                                     noverlap=window_size - 1)

    # absolute squared for power spectrum and taper averaging
    spec = np.real(ftr * ftr.conj()).mean(axis=1)[:, :, 0] # 1st Channel
    # amplitude spectrum
    ampls = np.sqrt(spec)

    fig, (ax1, ax2) = ppl.subplots(2, 1,
                                   sharex=True,
                                   gridspec_kw={"height_ratios": [1, 3]},
                                   figsize=(6, 6))

    ax1.set_title("Short Time Fourier Transform")
    ax1.plot(np.arange(signal.size) / fs, signal, c='cornflowerblue')
    ax1.set_ylabel('signal (a.u.)')

    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("frequency (Hz)")

    df = freqs[1] - freqs[0]
    # shift half a frequency bin
    extent = [0, len(signal) / fs, freqs[0] - df / 2, freqs[-1] - df / 2]
    # test also the plotting
    # scale with amplitude
    ax2.imshow(ampls.T,
               cmap='magma',
               aspect='auto',
               origin='lower',
               extent=extent,
               vmin=0,
               vmax=1.2 * A)

    # zoom into foi region
    ax2.set_ylim((foi[0], foi[-1]))

    # get the 'mappable' for the colorbar
    im = ax2.images[0]
    fig.colorbar(im, ax = ax2, orientation='horizontal',
                 shrink=0.7, pad=0.2, label='amplitude (a.u.)')

    # closest spectral indices to validate time-freq results
    freq_idx = []
    for frequency in signal_freqs:
        freq_idx.append(np.argmax(freqs >= frequency))

    # test amplitude normalization
    for idx, frequency in zip(freq_idx, signal_freqs):

        ax2.plot([0, len(signal) / fs],
                 [frequency, frequency],
                 '--',
                 c='0.5')

        # number of cycles with relevant
        # amplitude at the respective frequency
        cycle_num = (ampls[:, idx] > A / np.e).sum() / fs * frequency
        print(f'{cycle_num} cycles for the {frequency} band')
        # we have 2 times the cycles for each frequency (temporal neighbor)
        assert cycle_num > 2 * cycles
        # power should decay fast, so we don't detect more cycles
        assert cycle_num < 3 * cycles

    fig.tight_layout()

    # -------------------------
    # test multi-taper analysis
    # -------------------------

    taper = 'dpss'
    taper_opt = {'Kmax' : 10, 'NW' : 2}
    # the transforms have shape (nTime, nTaper, nFreq, nChannel)
    ftr2, freqs2 = mtmconvol.mtmconvol(signal,
                                       samplerate=fs, taper=taper, taper_opt=taper_opt,
                                       nperseg=window_size,
                                       noverlap=window_size - 1)

    spec2 = np.real((ftr2 * ftr2.conj()).mean(axis=1)[..., 0])
    # amplitude spectrum
    ampls2 = np.sqrt(spec2)

    fig, (ax1, ax2) = ppl.subplots(2, 1,
                                   sharex=True,
                                   gridspec_kw={"height_ratios": [1, 3]},
                                   figsize=(6, 6))

    ax1.set_title("Multi-Taper STFT")
    ax1.plot(np.arange(signal.size) / fs, signal, c='cornflowerblue')
    ax1.set_ylabel('signal (a.u.)')

    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("frequency (Hz)")

    # test also the plotting
    # scale with amplitude
    ax2.imshow(ampls2.T,
               cmap='magma',
               aspect='auto',
               origin='lower',
               extent=extent,
               vmin=0,
               vmax=1.2 * A)

    # zoom into foi region
    ax2.set_ylim((foi[0], foi[-1]))

    # get the 'mappable' for the colorbar
    im = ax2.images[0]
    fig.colorbar(im, ax = ax2, orientation='horizontal',
                 shrink=0.7, pad=0.2, label='amplitude (a.u.)')

    fig.tight_layout()

    for idx, frequency in zip(freq_idx, signal_freqs):

        ax2.plot([0, len(signal) / fs],
                 [frequency, frequency],
                 '--',
                 c='0.5')

    # for multi-taper stft we can't
    # check for the whole time domain
    # due to too much spectral broadening/smearing
    # so we just check that the maximum estimated
    # amplitude is within 10% boundsof the real amplitude

    assert 0.9 * A < ampls2.max() < 1.1 * A


def test_superlet():

    scalesSL = superlet.scale_from_period(1 / foi)

    # spec shape is nScales x nTime (x nChannels)
    spec = superlet.superlet(signal,
                             samplerate=fs,
                             scales=scalesSL,
                             order_max=20,
                             order_min=2,
                             c_1=1,
                             adaptive=False)
    # amplitude spectrum
    ampls = np.abs(spec)

    fig, (ax1, ax2) = ppl.subplots(2, 1,
                                   sharex=True,
                                   gridspec_kw={"height_ratios": [1, 3]},
                                   figsize=(6, 6))

    ax1.set_title("Superlet Transform")
    ax1.plot(np.arange(signal.size) / fs, signal, c='cornflowerblue')
    ax1.set_ylabel('signal (a.u.)')

    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("frequency (Hz)")
    extent = [0, len(signal) / fs, foi[0], foi[-1]]
    # test also the plotting
    # scale with amplitude
    ax2.imshow(ampls,
               cmap='magma',
               aspect='auto',
               extent=extent,
               origin='lower',
               vmin=0,
               vmax=1.2 * A)

    # get the 'mappable'
    im = ax2.images[0]
    fig.colorbar(im, ax = ax2, orientation='horizontal',
                 shrink=0.7, pad=0.2, label='amplitude (a.u.)')

    for idx, frequency in zip(freq_idx, signal_freqs):

        ax2.plot([0, len(signal) / fs],
                 [frequency, frequency],
                 '--',
                 c='0.5')

        # number of cycles with relevant
        # amplitude at the respective frequency
        cycle_num = (ampls[idx, :] > A / np.e).sum() / fs * frequency
        print(f'{cycle_num} cycles for the {frequency} band')
        # we have 2 times the cycles for each frequency (temporal neighbor)
        assert cycle_num > 2 * cycles
        # power should decay fast, so we don't detect more cycles
        assert cycle_num < 3 * cycles

    fig.tight_layout()


def test_wavelet():

    # get a wavelet function
    wfun = spywave.Morlet(10)
    scales = wfun.scale_from_period(1 / foi)

    # spec shape is nScales x nTime (x nChannels)
    spec = wavelet.wavelet(signal,
                           samplerate=fs,
                           scales=scales,
                           wavelet=wfun)
    # amplitude spectrum
    ampls = np.abs(spec)

    fig, (ax1, ax2) = ppl.subplots(2, 1,
                                   sharex=True,
                                   gridspec_kw={"height_ratios": [1, 3]},
                                   figsize=(6, 6))
    ax1.set_title("Wavelet Transform")
    ax1.plot(np.arange(signal.size) / fs, signal, c='cornflowerblue')
    ax1.set_ylabel('signal (a.u.)')

    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("frequency (Hz)")
    extent = [0, len(signal) / fs, foi[0], foi[-1]]

    # test also the plotting
    # scale with amplitude
    ax2.imshow(ampls,
               cmap='magma',
               aspect='auto',
               extent=extent,
               origin='lower',
               vmin=0,
               vmax=1.2 * A)

    # get the 'mappable'
    im = ax2.images[0]
    fig.colorbar(im, ax = ax2, orientation='horizontal',
                 shrink=0.7, pad=0.2, label='amplitude (a.u.)')

    for idx, frequency in zip(freq_idx, signal_freqs):

        ax2.plot([0, len(signal) / fs],
                 [frequency, frequency],
                 '--',
                 c='0.5')

        # number of cycles with relevant
        # amplitude at the respective frequency
        cycle_num = (ampls[idx, :] > A / np.e).sum() / fs * frequency
        print(f'{cycle_num} cycles for the {frequency} band')
        # we have at least 2 times the cycles for each frequency (temporal neighbor)
        assert cycle_num > 2 * cycles
        # power should decay fast, so we don't detect more cycles
        assert cycle_num < 3 * cycles

    fig.tight_layout()


def test_mtmfft():

    # superposition 40Hz and 100Hz oscillations A1:A2 for 1s
    f1, f2 = 40, 100
    A1, A2 = 5, 3
    tvec = np.arange(0, 1, 1 / 1000)

    signal = A1 * np.cos(2 * np.pi * f1 * tvec)
    signal += A2 * np.cos(2 * np.pi * f2 * tvec)

    # --------------------
    # -- test untapered --
    # --------------------

    # the transforms have shape (nTaper, nFreq, nChannel)
    ftr, freqs = mtmfft.mtmfft(signal, fs, taper=None)

    # with 1000Hz sampling frequency and 1000 samples this gives
    # exactly 1Hz frequency resolution ranging from 0 - 500Hz:
    assert freqs[f1] == f1
    assert freqs[f2] == f2

    # average over potential tapers (only 1 here)
    spec = np.real(ftr * ftr.conj()).mean(axis=0)
    amplitudes = np.sqrt(spec)[:, 0] # only 1 channel
    # our FFT normalisation recovers the signal amplitudes:
    assert np.allclose([A1, A2], amplitudes[[f1, f2]])

    fig, ax = ppl.subplots()
    ax.set_title(f"Amplitude spectrum {A1} x 40Hz + {A2} x 100Hz")
    ax.plot(freqs[:150], amplitudes[:150], label="No taper", lw=2)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('amplitude (a.u.)')

    # -------------------------
    # test multi-taper analysis
    # -------------------------

    taper_opt = {'Kmax' : 8, 'NW' : 1}
    ftr, freqs = mtmfft.mtmfft(signal, fs, taper="dpss", taper_opt=taper_opt)
    # average over tapers
    dpss_spec = np.real(ftr * ftr.conj()).mean(axis=0)
    dpss_amplitudes = np.sqrt(dpss_spec)[:, 0] # only 1 channel
    # check for amplitudes (and taper normalisation)
    assert np.allclose(dpss_amplitudes[[f1, f2]], [A1, A2], atol=1e-1)

    ax.plot(freqs[:150], dpss_amplitudes[:150], label="Slepian", lw=2)
    ax.legend()

    # -----------------
    # test kaiser taper (is boxcar for beta -> inf)
    # -----------------

    taper_opt = {'beta' : 2}
    ftr, freqs = mtmfft.mtmfft(signal, fs, taper="kaiser", taper_opt=taper_opt)
    # average over tapers (only 1 here)
    kaiser_spec = np.real(ftr * ftr.conj()).mean(axis=0)
    kaiser_amplitudes = np.sqrt(kaiser_spec)[:, 0] # only 1 channel
    # check for amplitudes (and taper normalisation)
    assert np.allclose(kaiser_amplitudes[[f1, f2]], [A1, A2], atol=1e-2)

    # -------------------------------
    # test all other window functions (which don't need a parameter)
    # -------------------------------

    for win in windows.__all__:
        taper_opt = {}
        # that guy isn't symmetric
        if win == 'exponential':
            continue
        # that guy is deprecated
        if win == 'hanning':
            continue
        try:
            ftr, freqs = mtmfft.mtmfft(signal, fs, taper=win, taper_opt=taper_opt)
            # average over tapers (only 1 here)
            spec = np.real(ftr * ftr.conj()).mean(axis=0)
            amplitudes = np.sqrt(spec)[:, 0] # only 1 channel
            # print(win, amplitudes[[f1, f2]])
            assert np.allclose(amplitudes[[f1, f2]], [A1, A2], atol=1e-3)
        except TypeError:
            # we didn't provide default parameters..
            pass

