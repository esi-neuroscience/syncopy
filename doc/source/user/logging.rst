.. _logging:

Trace Your Steps: Data Logs
===========================

An important feature of Syncopy fostering reproducibility is a ``log`` which gets attached to and propagated between all datasets. Suppose we have some :class:`~syncopy.SpectralData` and we want to know how we did arrive at these results, typing::

  spectral_data.log

Gives a output like this::

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


We see that from the creation of the original :class:`~syncopy.AnalogData` all steps needed to compute the new :class:`~syncopy.SpectralData` got recorded. In this example the spectra were computed via the multitapered FFT, with a spectral smoothing box (``tapsmofrq``) of 3Hz which required 5 Slepian tapers. The frequencies of interest (``foi``) range from 0Hz to 50Hz with 0.5Hz stepping and ``keeptrials`` was set to ``True``, meaning that this dataset contains the results for all trials separately.
