Quickstart with  Syncopy
============================

.. currentmodule:: syncopy

Here we want to quickly explore some standard analyses for analog data (e.g. MUA or LFP measurements), and how to do these in Syncopy. Explorative coding is best done interactively by using e.g. `Jupyter <https://jupyter.org>`_ or `IPython <https://ipython.org>`_. Note that for plotting also `matplotlib <https://matplotlib.org>`_ has to be installed. The following topics are covered here:

.. contents:: Topics covered
   :local:


Preparations
------------

To start with a clean slate, we will construct a synthetic damped harmonic with additive white noise.

.. hint::
   Further details about artifical data generatation can be found at the :ref:`synth_data` section.

.. literalinclude:: /scripts/qs_damped_harm.py


Data Object Inspection
----------------------

We can get some basic information about any Syncopy data set by just typing its name in an interpreter:

.. code-block:: python
		
   synth_data

which then gives a nicely formatted output:

.. code-block:: bash

   Syncopy AnalogData object with fields

            cfg : dictionary with keys ''
        channel : [2] element <class 'numpy.ndarray'>
      container : None
           data : 50 trials of length 1000.0 defined on [50000 x 3] float64 Dataset of size 1.14 MB
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

So we see that we indeed got 50 trials with 2 channels and 1000 samples each. Note that Syncopy per default **stores and writes all data on disc**, as this allows for seamless processing of larger than RAM datasets. The exact location and filename of a dataset in question is listed at the ``filename`` field. The standard location is the ``.spy`` directory created automatically in the users home directory. To change this and for more details please see :ref:`setup_env`.

.. hint::
   You can access each of the shown dataset fields separately using standard Python attribute access, e.g. ``synth_data.filename`` or ``synth_data.samplerate``.

		    
Time-Frequency Analysis
-----------------------

Syncopy groups analysis functionality into *meta-functions*, which in turn have various parameters selecting and controlling specific methods. In the case of spectral analysis the function to use is :func:`~syncopy.freqanalysis`.

Here we quickly want to showcase two important methods for (time-)frequency analysis: (multi-tapered) FFT and Wavelet analysis.

Multitapered Fourier Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Multitaper methods <https://en.wikipedia.org/wiki/Multitaper>`_ allow for frequency smoothing of Fourier spectra. Syncopy implements the standard `Slepian/DPSS tapers <https://en.wikipedia.org/wiki/Window_function#DPSS_or_Slepian_window>`_ and provides a convenient parameter, the *taper smoothing frequency* ``tapsmofrq`` to control the amount of spectral smoothing in Hz. To perform a multi-tapered Fourier analysis with 3Hz spectral smoothing, we simply do:

.. code-block::

   spectra = spy.freqanalsysis(synth_data, method='mtmfft', foilim=[0, 50], taper='dpss', tapsmofrq=3)

The parameter ``foilim`` controls the *frequencies of interest  limits*, so in this case we are interested in the range 0-50Hz. Starting the computation interactively will show additional information::

  Syncopy <validate_taper> INFO: Using 5 taper(s) for multi-tapering

informing us, that for this dataset a spectral smoothing of 3Hz required 5 Slepian tapers.

Excurs: Logging
"""""""""""""""

An important feature of Syncopy fostering reproducibility is a ``log`` which gets attached to and propagated between datasets. Typing::

  spectra.log

Gives the following (similar) output::

  |=== user@machine: Thu Feb  3 17:05:59 2022 ===|

	__init__: created AnalogData object

  |=== user@machine: Thu Feb  3 17:12:11 2022 ===|

	__init__: created SpectralData object

  |=== user@machine: Thu Feb  3 17:12:11 2022 ===|

	definetrial: updated trial-definition with [50 x 3] element array

  |=== user@machine: Thu Feb  3 17:12:11 2022 ===|

	write_log: computed mtmfft_cF with settings
	method = mtmfft
	output = pow
	keeptapers = False
	keeptrials = True
	polyremoval = None
	pad_to_length = None
	foi = [ 0.   0.5  1.   1.5  2.   2.5, ..., 47.5 48.  48.5 49.  49.5 50. ]
	taper = dpss
	nTaper = 5
	tapsmofrq = 3


We see that from the creation of the original :class:`~syncopy.AnalogData` all steps needed to compute our new :class:`~syncopy.SpectralData` got recorded.

To quickly have something for the eye we can plot the power spectrum using the generic :func:`syncopy.singlepanelplot`::

  spectra.singlepanelplot()

.. image:: mtmfft_spec.png
   :height: 250px

The originally very sharp harmonic peak around 30Hz got widened to about 3Hz, for all other frequencies we have the expected flat white noise floor.
