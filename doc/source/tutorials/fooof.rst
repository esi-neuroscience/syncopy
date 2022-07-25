Using FOOOF from syncopy
========================

Syncopy supports parameterization of neural power spectra using
the `Fitting oscillations & one over f` (`FOOOF <https://github.com/fooof-tools/fooof>`_
) method described in the following publication (`DOI link <https://doi.org/10.1038/s41593-020-00744-x>`):

`Donoghue T, Haller M, Peterson EJ, Varma P, Sebastian P, Gao R, Noto T, Lara AH, Wallis JD,
Knight RT, Shestyuk A, & Voytek B (2020). Parameterizing neural power spectra into periodic
and aperiodic components. Nature Neuroscience, 23, 1655-1665.
DOI: 10.1038/s41593-020-00744-x`

The FOOOF method requires that you have your data in a Syncopy `AnalogData` instance,
and applying FOOOF can be seen as a post-processing of an MTMFFT.


Generating Example Data
-----------------------

Let us first prepare
suitable data. FOOOF will typically be applied to trial-averaged data, as the method is
quite sensitive to noise, so we generate an example data set consisting of 200 trials and
a single channel here:

.. code-block:: python
    :linenos:

    import numpy as np
    from syncopy import freqanalysis, get_defaults
    from syncopy.tests.synth_data import AR2_network, phase_diffusion

    def get_signal(nTrials=200, nChannels = 1):
        nSamples = 1000
        samplerate = 1000
        ar1_part = AR2_network(AdjMat=np.zeros(1), nSamples=nSamples, alphas=[0.9, 0], nTrials=nTrials)
        pd1 = phase_diffusion(freq=30., eps=.1, fs=samplerate, nChannels=nChannels, nSamples=nSamples, nTrials=nTrials)
        pd2 = phase_diffusion(freq=50., eps=.1, fs=samplerate, nChannels=nChannels, nSamples=nSamples, nTrials=nTrials)
        signal = ar1_part + .8 * pd1 + 0.6 * pd2
        return signal

    dt = get_signal()

Let's have a look at the signal in the time domain first:

.. code-block:: python
    :linenos:

    dt.singlepanelplot(trials = 0)

.. image:: ../_static/fooof_signal_time.png

Since FOOOF works on the power spectrum, we can perform an `mtmfft` and look at the results to get
a better idea of how our data look in the frequency domain. The `spec_dt` data structure we obtain is
of type `syncopy.SpectralData`, and can also be plotted:

.. code-block:: python
    :linenos:

    cfg = get_defaults(freqanalysis)
    cfg.method = "mtmfft"
    cfg.taper = "hann"
    cfg.select = {"channel": 0}
    cfg.keeptrials = False
    cfg.output = "pow"
    cfg.foilim = [10, 100]

    spec_dt = freqanalysis(cfg, dt)
    spec_dt.singlepanelplot()


.. image:: ../_static/fooof_signal_spectrum.png


Running FOOOF
-------------

Now that we have seen the data, let us start FOOOF. The FOOOF method is accessible
from the `freqanalysis` function.

When running FOOOF, it

* **fooof**: the full fooofed spectrum
* **fooo_aperiodic**: the aperiodic part of the spectrum
* **fooof_peaks**: the detected peaks, with Gaussian fit to them

.. code-block:: python
    :linenos:

    cfg.out = 'fooof'
    spec_dt = freqanalysis(cfg, dt)
    spec_dt.singlepanelplot()

.. image:: ../_static/fooof_out_first_try.png

Knowing what your data and the FOOOF results like is important, because typically
you will have to fine tune the FOOOF method to get the results you are interested in.
This can be achieved by using the `fooof_opt` parameter to `freqanalyis`.

From the results above, we see that some peaks were detected that we feel are noise.
Increasing the minimal peak width is one method to exclude them:

.. code-block:: python
    :linenos:

    cfg.fooof_opt = {'peak_width_limits': (6.0, 12.0), 'min_peak_height': 0.2}
    spec_dt = freqanalysis(cfg, tf)
    spec_dt.singlepanelplot()

Once more, look at the FOOOFed spectrum:

.. image:: ../_static/fooof_out_tuned.png

