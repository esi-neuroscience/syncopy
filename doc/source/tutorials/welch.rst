Running Welch's method for the estimation of power spectra in Syncopy
=====================================================================

Welch's method for the estimation of power spectra based on time-averaging over short, modified periodograms
is described in the following publication:

`Welch, P. (1967). The use of fast Fourier transform for the estimation of power spectra:
a method based on time averaging over short, modified periodograms.
IEEE Transactions on Audio and Electroacoustics, 15(2), 70-73.
https://doi.org/10.1109/TAU.1967.1161901`

Generating Example Data
-----------------------

Let us first prepare suitable data, we use white noise here:

.. code-block:: python
    :linenos:

    import syncopy as spy
    import syncopy.tests.synth_data as synth_data

    wn = synth_data.white_noise(nTrials=2, nChannels=3, nSamples=20000, samplerate=1000)

The return value `wn` is of type :class:`~syncopy.AnalogData` and contains 2 trials and 3 channels,
each consisting of 20 seconds of white noise: 20000 samples at a sample rate of 1000 Hz.

We now create a config for running Welch's method and call `freqanalysis` with it:

.. code-block:: python
    :linenos:

    cfg = spy.get_defaults(spy.freqanalysis)
    cfg.method = "welch"
    cfg.t_ftimwin = 0.5  # Window length in seconds.
    cfg.toi = 0.0        # Overlap between windows, 0.5 = 50 percent overlap.

    welch_res = spy.freqanalysis(cfg, wn)


Let's inspect the resulting `SpectralData` instance by looking at its dimensions, and then visualize it:

.. code-block:: python
    :linenos:

    welch_res.dimord      # ('time', 'taper', 'freq', 'channel',)
    welch_res.data.shape  # (2, 1, ?, 3)

The `time` axis contains two entries, one per trial, because by default there is no trial averaging (`cfg.keeptrials` is `True`). With trial averaging,
there would only be a single entry here.

The `taper` axis will always have size 1 for Welch, even for multi-tapering, as taper averaging must be active for Welch (`cfg.keeptapers` must be `False`), as explained in the function documentation.

The size of the frequency axis, i.e., the frequency resolution, depends on the length of the input windows and is thus a function of the input signal, `cfg.t_ftimwin`, `cfg.toi`, and potentially other settings (like a `foilim`).

The channels are left as is.

We can also visualize the power spectrum. Here we select the first of the two trials:

.. code-block:: python
    :linenos:

    _, ax = welch_res.singlepanelplot(trials=0, logscale=False)
    ax.set_title("Welch result")
    ax.set_ylabel("Power")
    ax.set_xlabel("Frequency")



This concludes the tutorial on using FOOOF from Syncopy.