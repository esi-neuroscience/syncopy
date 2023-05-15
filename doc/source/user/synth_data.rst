.. _synthdata:

==============
Synthetic Data
==============

For testing and educational purposes it is always good to work with synthetic data. Syncopy brings its own suite of synthetic data generators, but it is also possible to devise your own synthetic data using standard `NumPy <https://numpy.org>`_.

.. contents::
   :local:


Built-in Generators
===================

These functions return a multi-trial :class:`~syncopy.AnalogData` object representing multi-channel time series data:

.. autosummary::

   syncopy.synthdata.harmonic
   syncopy.synthdata.white_noise
   syncopy.synthdata.red_noise
   syncopy.synthdata.linear_trend
   syncopy.synthdata.phase_diffusion
   syncopy.synthdata.ar2_network

With the help of basic arithmetical operations we can combine different synthetic signals to arrive at more complex ones. Let's look at an example::

  import syncopy as spy

  # set up cfg
  cfg = spy.StructDict()
  cfg.nTrials = 40
  cfg.samplerate = 500
  cfg.nSamples = 500
  cfg.nChannels = 5

  # start with a simple 60Hz harmonic
  sdata = spy.synthdata.harmonic(freq=60, cfg=cfg)

  # add some strong AR(1) process as surrogate 1/f
  sdata = sdata + 5 * spy.synthdata.red_noise(alpha=0.95, cfg=cfg)

  # plot all channels for a single trial
  sdata.singlepanelplot(trials=10)

  # compute spectrum and plot trial average of 2 channels
  spec = spy.freqanalysis(sdata, keeptrials=False)
  spec.singlepanelplot(channel=[0, 2], frequency=[0,100])

.. image:: /_static/synth_data1.png
   :height: 300px

.. image:: /_static/synth_data1_spec.png
   :height: 300px

.. _gen_synth_recipe:

Phase diffusion
---------------

A diffusing phase can be modeled by adding white noise :math:`\xi(t)` to a fixed angular frequency:

.. math::
   \omega(t) = \omega + \epsilon \xi(t),

with the instantaneous frequency :math:`\omega(t)`.

Integration then yields the phase trajectory:

.. math::
   \phi(t) = \int_0^t \omega(t) = \omega t + \epsilon W(t).

Here :math:`W(t)` being the `Wiener process <https://en.wikipedia.org/wiki/Wiener_process>`_, or simply a one dimensional diffusion process. Note that for the trivial case :math:`\epsilon = 0`, so no noise got added, the phase describes a linear constant motion with the `phase velocity` :math:`\omega = 2\pi f`. This is just a harmonic oscillation with frequency :math:`f`. Finally, by wrapping the phase trajectory into a :math:`2\pi` periodic `waveform` function, we arrive at a time series (or signal). The simplest waveform is just the cosine, so we have:

.. math::
   x(t) = cos(\phi(t))

This is exactly what the :func:`~syncopy.synthdata.phase_diffusion` function provides.

Phase diffusing models have some interesting properties, let's have a look at the power spectrum::

  import syncopy as spy

  cfg = spy.StructDict()
  cfg.nTrials = 250
  cfg.nChannels = 2
  cfg.samplerate = 500
  cfg.nSamples = 2000

  # harmonic frequency is 60Hz, phase diffusion strength is 0.01
  signals = spy.synthdata.phase_diffusion(freq=60, eps=0.01, cfg=cfg)

  # add harmonic frequency with 20Hz, there is not phase diffusion
  signals += spy.synthdata.harmonic(freq=20, cfg=cfg)

  # freqanalysis without tapering and absolute power
 
  cfg_freq = spy.StructDict()
  cfg_freq.keeptrials = False
  cfg_freq.foilim = [2, 100]
  cfg_freq.output = 'abs'
  cfg_freq.taper = None
  
  spec = spy.freqanalysis(signals, cfg=cfg_freq)
  spec.singlepanelplot(channel=0)

.. image:: /_static/synth_data_pdiff_spec.png
   :height: 300px

We see a natural (no tapering) spectral broadening for the phase diffusing signal at 60Hz, reflecting the fluctuations in instantaneous frequency.

General Recipe for custom Synthetic Data
=========================================

We can easily create custom synthetic datasets using basic `NumPy <https://numpy.org>`_ functionality and Syncopy's :class:`~syncopy.AnalogData`.

To create a synthetic timeseries data set follow these steps:

- write a function which returns a single trial as a 2d-:class:`~numpy.ndarray`  with desired shape ``(nSamples, nChannels)``
- collect all the trials into a Python ``list``, for example with a list comprehension or simply a for loop
- Instantiate an  :class:`~syncopy.AnalogData` object by passing this list holding the trials as ``data`` and set the desired ``samplerate``

In (pseudo-)Python code:

.. code-block:: python

   def generate_trial(nSamples, nChannels):

	trial = .. something fancy ..

	# These should evaluate to True
	isinstance(trial, np.ndarray)
	trial.shape == (nSamples, nChannels)

	return trial

   # collect the trials
   nSamples = 1000
   nChannels = 2
   nTrials = 100
   trls = []

   for _ in range(nTrials):
       trial = generate_trial(Samples, nChannels)
       # manipulate further as needed, e.g. add a constant
       trial += 3
       trls.append(trial)

   # instantiate syncopy data object
   my_fancy_data = spy.AnalogData(data=trls, samplerate=my_samplerate)

.. note::
    The same recipe can be used to generally instantiate Syncopy data objects from NumPy arrays.

.. note::
    Syncopy data objects also accept Python generators as ``data``, allowing to stream
    in trial arrays one by one. In effect this allows creating datasets which are larger
    than the systems memory. This is also how the build in generators of ``syncopy.synthdata`` (see above) work under the hood.


Example: Noisy Harmonics
---------------------------

Let's create two harmonics and add some white noise to it:

.. literalinclude:: /scripts/synth_data1.py

Here we first defined the number of trials (``nTrials``) and then the number of samples (``nSamples``) and channels (``nChannels``) per trial. With a sampling rate of 500Hz and 1000 samples this gives us a trial length of two seconds. The function ``generate_noisy_harmonics`` adds a 20Hz harmonic on the 1st channel, a 50Hz harmonic on the 2nd channel and white noise to all channels, Every trial got collected into a Python ``list``, which at the last line was used to initialize our :class:`~syncopy.AnalogData` object ``synth_data``. Note that data instantiated that way always has a default trigger offset of -1 seconds.

Now we can directly run a multi-tapered FFT analysis and plot the power spectra of all 2 channels:

.. code-block:: python

   spectrum = spy.freqanalysis(synth_data, foilim=[0,80], tapsmofrq=2, keeptrials=False)
   spectrum.singlepanelplot()


.. image:: /_static/synth_data_spec.png
   :height: 300px

As constructed, we have two harmonic peaks at the respective frequencies (20Hz and 50Hz) and the white noise floor on all channels.
