Resample data with Syncopy
==========================

Changing the sampling rate of a dataset is a common tasks in digital signal processing. Syncopy offers simple *downsampling* (decimation) with or without explicit low pass filtering, and the numerically more expensive *resampling* to arbitrary new sample rates.

.. Note::
   Our friends at FieldTrip also have a nice tutorial about resampling `here <https://www.fieldtriptoolbox.org/faq/resampling_lowpassfilter>`_

Synthetic Data
--------------

To start with a clean slate, let's construct a synthetic signal with two harmonics,
one at 100Hz and one at 1300Hz:

.. code-block:: python
    :linenos:

    # Import package
    import syncopy as spy
    from syncopy.tests import synth_data

    # basic dataset properties
    nTrials = 20
    samplerate = 5000  # in Hz

    # add a harmonic with 100Hz
    adata = synth_data.harmonic(nTrials, freq=100, samplerate=samplerate)

    # add another harmonic with 1300Hz
    adata += synth_data.harmonic(nTrials, freq=1300, samplerate=samplerate)

Note that our *original sampling rate* is 5000Hz, so both harmonics are well sampled as can be seen in the power spectrum::

  # compute the trial averaged spectrum and plot
  spec = spy.freqanalysis(adata, keeptrials=False)
  spec.singlepanelplot(channel=0)

.. image:: res_orig_spec.png

Downsampling
------------

Suppose we want to downsample our signal to 1kHz. The *original sampling rate* here is an exact integer multiple of our *new sampling rate*: :math:`5 \times 1000Hz = 5000Hz` Hence, we can use direct *decimation* or downsampling. This really is only a fancy term for taking every `nth` data point, where `n` is the integer division factor, which is 5 in our case. In Syncopy we can directly use ``method='downsample'``::

  ds_adata = spy.resampledata(adata, method='downsample', resamplefs=1000)

Let's have a look at the new power spectrum::
  
  ds_spec = spy.freqanalysis(adata, keeptrials=False)
  ds_spec.singlepanelplot(channel=0)

.. image:: res_ds_spec.png

What happened? First we have to note that the frequency axis now goes from 0Hz to only 500Hz, the *new Nyquist frequency* which is half of our new sample rate. We still see our expected peak at 100Hz, but there is also another one at 300Hz even though our signal never contained such oscillations! This phenomenon is called `aliasing <https://en.wikipedia.org/wiki/Aliasing>`_, basically meaning that frequencies which are present in the signal yet are higher than the Nyquist limit (:math:`1300 Hz > 500Hz`) "wrap around" and re-appear as spurious low-frequency components. The formula for alias frequencies is:

.. math::

   f_{alias} = \left |\frac{f_{sample}}{n} - f_{orig}\right | 

where :math:`f_{sample}` is the original sampling frequency, :math:`n` is the integer decimation factor and :math:`f_{orig}` is the original frequency. Plugging in the values we have gives: :math:`|5000Hz / 5 - 1300Hz| = 300Hz`. So this 300Hz peak is our old 1300Hz peak, but shifted by 1000Hz.

.. note::
   To calculate the alias frequencies for original frequencies which after one application of the formula above are still outside the new frequency range, just re-apply the formula. For example say :math:`f_{orig} = 1700Hz`, after one round we get :math:`|1000Hz - 1700Hz| = 700Hz` which is still outside our spectral range of 0-500Hz. A 2nd iteration then yields :math:`1000Hz - 700Hz = 300Hz`, meaning that both 1300Hz and 1700Hz components get aliased by the same frequency, try it yourself ;)

Low-pass filtering
^^^^^^^^^^^^^^^^^^

To circumvent the problem of aliasing, application of a so-called *anti-alias-filter* is advisable. There is nothing special with that filter as such, basically every low-pass filter will work. The critical step is to filter out all frequencies which are greater than the *new Nyquist frequency*. Filtering out more than those does not introduce additional artifacts, however setting the cut-off to high gets only partially rid of the aliasing. In Syncopy we enforce that if a filtering step is requested by setting the cut-off frequency ``lpfreq`` parameter, it has to be maximally the new Nyquist. So trying::

  ds_data2 = spy.resampledata(adata, method='downsample', resamplefs=1000, lpfreq=600)

throws::

  >>> SPYValueError: Invalid value of `lpfreq`: '600'; expected value to be greater or equals 0 and less or equals 500.0
  
because 600Hz is still bigger than the new Nyquist of :math:`1000Hz / 2 = 500Hz`. But this here will work just fine and results in the expected spectrum::

  ds_data2 = spy.resampledata(adata, method='downsample', resamplefs=1000, lpfreq=400)
  
