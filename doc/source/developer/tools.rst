Tools for Developing Syncopy
============================
Coming soon...

Input Parsing and Error Checking
--------------------------------

.. autosummary::
   :toctree: _stubs    
   
   syncopy.shared.parsers.array_parser
   syncopy.shared.parsers.data_parser
   syncopy.shared.parsers.filename_parser
   syncopy.shared.parsers.io_parser
   syncopy.shared.parsers.scalar_parser

Decorators
----------

.. autosummary::
   :toctree: _stubs    
   
   syncopy.shared.kwarg_decorators.unwrap_cfg
   syncopy.shared.kwarg_decorators.unwrap_select
   syncopy.shared.kwarg_decorators.unwrap_io


Writing A New Analysis Routine
------------------------------

Any analysis routine that operates on Syncopy data is always structured in three
(hierarchical) parts:

1. A numerical function based only on NumPy/SciPy that works on a
   :class:`numpy.ndarray` and returns a :class:`numpy.ndarray`. 
2. A wrapper class that handles output initialization, potential
   parallelization and post-computation cleanup. The class should be based on the
   abstract class :class:`syncopy.shared.computational_routine.ComputationalRoutine`
3. Another wrapping metafunction handling method selection, parameterization and
   error checking is then provided user interaction.

An example of this type of structure is the multi-taper fourier analysis. The
corresponding stages here are

1. Numerical function: :func:`syncopy.specest.mtmfft`
2. Wrapper class: :class:`syncopy.specest.MultiTaperFFT`
3. Metafunction: :func:`syncopy.freqanalysis` 

.. image:: ../_static/ComputationalRoutine.png

For a detailed walk-through explaining the intricacies of writing an analysis
routine, please refer to the :doc:`compute_kernels`.
