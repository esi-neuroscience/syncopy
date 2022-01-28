Quickstart with  Syncopy
============================

.. currentmodule:: syncopy

Here we want to quickly explore some standard analyses for analog data (e.g. MUA or LFP measurements), and how to do these in Syncopy. Explorative coding is best done interactively by using e.g. `Jupyter <https://jupyter.org>`_ or `IPython <https://ipython.org>`_. Note that for plotting also `matplotlib <https://matplotlib.org>`_ has to be installed. The following topics are covered here:

- :ref:`synth_data`
  
.. _synth_data:
   
Synthetic Data
--------------

For testing and demonstrational purposes it is always good to work with synthetic data. In Syncopy we can easily create a synthetic dataset using basic `NumPy <https://numpy.org>`_ functionality and Syncopy's :class:`~syncopy.AnalogData`. To start simple we create two harmonics and add some white noise to it:

.. literalinclude:: /scripts/qs_synth_data1.py


Here we first defined the number of trials and then the number of samples and channels per trial. With a sampling rate of 500Hz and 1000 samples this gives us a trial length of two seconds. After creating the two harmonics we sampled Gaussian white noise for each trial, and added the 30Hz harmonic on the 1st channel and the 42Hz harmonic on the 2nd channel. With this the 3rd channel is left with only the noise. Every trial got collected into a Python ``list``, which at the last line was used to initialize our :class:`~syncopy.AnalogData` object. Note that synthetic data always is created with a default trigger offset of -1 seconds.

We can get some basic information about any Syncopy data set by just typing its name in an interpreter:

.. code-block:: python
		
   data

which then gives a nicely formatted output:

.. code-block:: bash

   Syncopy AnalogData object with fields

            cfg : dictionary with keys ''
        channel : [3] element <class 'numpy.ndarray'>
      container : None
           data : 50 trials of length 1000.0 defined on [50000 x 3] float64 Dataset of size 1.14 MB
         dimord : time by channel
       filename : /home/whir/.spy/spy_910e_572582c9.analog
           mode : r+
     sampleinfo : [50 x 2] element <class 'numpy.ndarray'>
     samplerate : 500.0
            tag : None
           time : 50 element list
      trialinfo : [50 x 0] element <class 'numpy.ndarray'>
         trials : 50 element iterable

   Use `.log` to see object history

So we see that we indeed got 50 trials and 3 channels. To quickly plot some raw data we can use the :meth:`~syncopy.AnalogData.show` method to make various selections:

.. code-block:: python

   import matplotlib.pyplot as ppl

   # the selection
   chan1and2 = data.show(trials=[3], channels=['channel1', 'channel2'])

   # the plotting
   ppl.plot(data.time[0], chan1and2, label=['channel1', 'channel2'])
   # zoom into first 250ms
   ppl.xlim((-1, -.75))
   # add a xlabel
   ppl.xlabel("time (s)")
   # add the legend
   ppl.legend()

Here we are looking at the first 250ms of the 4th trial of channels 1 and 2. After inputting the above code you should see a plot akin to this one here:

.. image:: synth_data_plot.png
   :height: 400px
   :align: left
   
|
	 
We see two noisy oscillatory time-series, with channel 2 being faster as channel 1 (42Hz vs. 30Hz) as expected.

Time-Frequency Analysis
-----------------------





