Tools for developing Syncopy
============================
Some profoundly insightful text here...



Input parsing and error checking
--------------------------------

.. autosummary::
   :toctree: _stubs    
   
   syncopy.shared.parsers.io_parser
   syncopy.shared.parsers.scalar_parser
   syncopy.shared.parsers.array_parser
   syncopy.shared.parsers.get_defaults   


Writing A New Analysis Routine
------------------------------

Any analysis routine that operates on Syncopy data is always structured in three
(hierarchical) parts:

1. A numerical function based on NumPy/SciPy only that works on a
   :class:`numpy.ndarray` and returns a :class:`numpy.ndarray`. 
2. A wrapper class that handles output initializiation, potential
   parallelization and post-computation cleanup. This should be based on the
   abstract class :class:`syncopy.shared.computational_routine.ComputationalRoutine`
3. Another wrapping metafunction handling method selection, parameterization and
   error checking is then provided for the users.

An example for this type of structure is the multi-taper fourier analysis. The
corresponding stages here are

1. Numerical function: :func:`syncopy.specest.mtmfft`
2. Wrapper class: :class:`syncopy.specest.MultiTaperFFT`
3. Metafunction: :func:`syncopy.freqanalysis` 

For a detailed walk-through explaining the intricacies of writing an analysis
routine, please refer to the :doc:`compute_kernels`.

