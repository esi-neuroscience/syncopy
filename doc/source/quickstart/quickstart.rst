.. _quick_start:

************************
Quickstart with Syncopy
************************

Here we want to quickly explore some standard analyses for analog data (e.g. MUA or LFP measurements), and how to do these in Syncopy. Explorative coding is best done interactively by using e.g. `Jupyter <https://jupyter.org>`_ or `IPython <https://ipython.org>`_. Note that for plotting also `matplotlib <https://matplotlib.org>`_ has to be installed.

.. contents:: Topics covered
   :local:

.. note::
   Installation of Syncopy itself is covered in :doc:`here </setup>`.


Preparations
============

To start with a clean slate, let's construct a synthetic dataset consisting of a damped 30Hz harmonic and additive white noise:

.. literalinclude:: /quickstart/damped_harm.py

With this we have a dataset of type :class:`~syncopy.AnalogData`, which is intended for holding time-series data like electrophys. measurements. Let's have a look at a small snippet of the 1st trial::

  data.singlepanelplot(trials=0, latency=[0, 0.5])

.. image:: damped_signals.png
   :height: 220px

By construction, we made the (white) noise of the same strength as the signal, hence by eye the oscillations present in channel1 are hardly visible. The parameter ``latency`` defines a time-interval selection here.

.. hint::
   How to plot and work with subsets of Syncopy data is described in :ref:`selections`.

To recap: we have generated a synthetic dataset white noise on both channels, and channel1 additionally carries the damped harmonic signal.

.. hint::
   Further details about artificial data generation can be found at the :ref:`synthdata` section.


Data Object Inspection
======================

We can get some basic information about any Syncopy dataset by just typing its name in an interactive Python interpreter:

.. code-block:: python

   data

which gives nicely formatted output:

.. code-block:: bash

   Syncopy AnalogData object with fields

            cfg : dictionary with keys ''
        channel : [2] element <class 'numpy.ndarray'>
      container : None
           data : 50 trials of length 1000.0 defined on [50000 x 2] float32 Dataset of size 1.14 MB
         dimord : time by channel
       filename : /xxx/xxx/.spy/spy_910e_572582c9.analog
           mode : r+
     sampleinfo : [50 x 2] element <class 'numpy.ndarray'>
     samplerate : 500.0
            tag : None
           time : 50 element list
      trialinfo : [50 x 0] element <class 'numpy.ndarray'>
         trials : 50 element iterable

   Use `.log` to see object history


So we see that we indeed got 50 trials with 2 channels and 1000 samples each. Note that Syncopy per default **stores and writes all data on disk**, as this allows for seamless processing of **larger than memory** datasets. The exact location and filename of a dataset in question is listed at the ``filename`` field. The standard location is the ``.spy`` directory created automatically in the user's home directory. To change this and for more details please see :ref:`setup_env`.

.. hint::
   You can access each of the shown meta-information fields separately using standard Python attribute access, e.g. ``data.filename`` or ``data.samplerate``.


Spectral Analysis
=================

Syncopy groups analysis functionality into *meta-functions*, which in turn have various parameters selecting and controlling specific methods. In the case of spectral analysis the function to use is :func:`~syncopy.freqanalysis`.

Here we quickly want to showcase two important methods for (time-)frequency analysis: (multi-tapered) FFT and Wavelet analysis.

.. _mtmfft:

Multitapered Fourier Analysis
------------------------------

`Multitaper methods <https://en.wikipedia.org/wiki/Multitaper>`_ allow for frequency smoothing of Fourier spectra. Syncopy implements the standard `Slepian/DPSS tapers <https://en.wikipedia.org/wiki/Window_function#DPSS_or_Slepian_window>`_ and provides a convenient parameter, the *taper smoothing frequency* ``tapsmofrq`` to control the amount of one-sided spectral smoothing in Hz. To perform a multi-tapered Fourier analysis with 2Hz spectral smoothing (1Hz two sided), we simply do:

.. code-block::

   fft_spectra = spy.freqanalysis(data, method='mtmfft', foilim=[0, 60], tapsmofrq=1)

The parameter ``foilim`` controls the *frequencies of interest  limits*, so in this case we are interested in the range 0-60Hz. Starting the computation interactively will show additional information::

  Syncopy <validate_taper> INFO: Using 3 taper(s) for multi-tapering

informing us, that for this dataset a total spectral smoothing of 2Hz required 3 Slepian tapers.

The resulting new dataset ``fft_spectra`` is of type :class:`syncopy.SpectralData`, which is the general datatype storing the results of a time-frequency analysis.

.. hint::
   Try typing ``fft_spectra.log`` into your interpreter and have a look at :doc:`Trace Your Steps: Data Logs </user/logging>` to learn more about Syncopy's logging features

To quickly have something for the eye we can compute the trial average and plot the power spectrum using the generic :func:`syncopy.singlepanelplot`::

  # compute trial average
  fft_avg = spy.mean(fft_spectra, dim='trials')

  # plot frequency range between 10Hz and 50Hz
  fft_avg.singlepanelplot(frequency=[10, 50])

.. image:: mtmfft_spec.png
   :height: 260px

We clearly see a smoothed spectral peak at 30Hz, channel 2 just contains the flat white noise floor. Comparing with the signals plotted in the time domain above, we see the power of the frequency representation of an oscillatory signal.

The related short time Fourier transform can be computed via ``method='mtmconvol'``, see :func:`~syncopy.freqanalysis` for more details and examples.

.. note::
   Have a look at :ref:`workflow` to get an overview about data processing principles with Syncopy

Wavelet Analysis
----------------

`Wavelet Analysis <https://en.wikipedia.org/wiki/Continuous_wavelet_transform>`_, especially with `Morlet Wavelets <https://en.wikipedia.org/wiki/Morlet_wavelet>`_, is a well established method for time-frequency analysis. For each frequency of interest (``foi``), a Wavelet function gets convolved with the signal yielding a time dependent cross-correlation. By (densely) scanning a range of frequencies, a continuous time-frequency representation of the original signal can be generated.

In Syncopy we can compute the Wavelet transform by calling :func:`~syncopy.freqanalysis` with the ``method='wavelet'`` argument::

  # define frequencies to scan
  fois = np.arange(10, 60, step=2) # 2Hz stepping
  wav_spectra = spy.freqanalysis(data,
                                 method='wavelet',
				 foi=fois,
				 parallel=True,
				 keeptrials=False)

Here we used two additional parameters supported by every Syncopy analysis method:

- ``parallel=True`` invokes Syncopy's parallel processing engine
- ``keeptrials=False`` triggers trial averaging

.. hint::

   If parallel processing is unavailable, have a look at :ref:`install_acme`

To quickly inspect the results for each channel we can use::

  wav_spectra.multipanelplot()

.. image:: wavelet_spec.png
   :height: 250px

Again, we see a strong 30Hz signal in the 1st channel, and channel 2 is devoid of any rhythms. However, in contrast to the ``method='mtmfft'`` call, now we also get information along the time axis. The dampening of the 30Hz harmonic over time in channel 1 is clearly visible.

An improved method, the superlet transform, providing super-resolution time-frequency representations can be computed via ``method='superlet'``, see :func:`~syncopy.freqanalysis` for more details.

