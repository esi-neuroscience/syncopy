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

