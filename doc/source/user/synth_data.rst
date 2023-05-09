.. _synthdata:
   
Synthetic Data
==============

For testing and educational purposes it is always good to work with synthetic data. Syncopy brings its own suite of synthetic data generators, but it is also possible to devise your own synthetic data using standard `NumPy <https://numpy.org>`_.

.. contents::
   :local:


Built-in Generators
-------------------

Using syncopy's homegrown synthetic data generators is straightforward, they all reside inside the ``syncopy.synthdata`` module. With the help of basic arithmetical operations different scenarios are possible. Let's look at an example::

  import syncopy as spy

  cfg = spy.StructDict()
  cfg.nTrials = 40
  cfg.samplerate = 500
  cfg.nSamples = 500
  cfg.nChannels = 5
  
  sdata = spy.synthdata.harmonic(freq=10, cfg)


These generators return a multi-trial :class:`~syncopy.AnalogData` object representing multi-channel time series data:

.. autosummary::

   syncopy.synthdata.harmonic
   syncopy.synthdata.linear_trend
   syncopy.synthdata.phase_diffusion
   syncopy.synthdata.AR2_network
   syncopy.synthdata.white_noise


.. _gen_synth_recipe:

General Recipe
--------------

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
    



