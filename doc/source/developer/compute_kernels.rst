:orphan:

.. contents::
    Contents
    :local:

.. _cf: _stubs/syncopy.shared.computational_routine.ComputationalRoutine.html#syncopy.shared.computational_routine.ComputationalRoutine.computeFunction
.. |cf|	replace:: :meth:`computeFunction`

.. currentmodule:: syncopy.shared.computational_routine

Design Guide: Syncopy Compute Classes
=====================================
A compute class represents the centerpiece of a Syncopy analysis routine.
The abstract base class :class:`ComputationalRoutine` is the concrete
realization of a general-purpose computing object. This class provides a
blueprint for implementing algorithmic strategies in Syncopy. Every
computational method in Syncopy consists of a core routine, the |cf|_,
which can be executed either sequentially or fully parallel. To unify
common instruction sequences and minimize code redundancy, Syncopy's
:class:`ComputationalRoutine` manages all pre- and post-processing steps
necessary during preparation and after termination of a calculation. This
permits developers to focus exclusively on the implementation of the actual
algorithmic details when including a new computational method in Syncopy.

Designing a |cf|_
-----------------
For enabling :class:`ComputationalRoutine` to perform all required
computational management tasks, a |cf|_ has to satisfy a few basic
requirements. Syncopy leverages a hierarchical parallelization paradigm
whose low-level foundation is represented by trial-based parallelism (its
open-ended higher levels may constitute by-object, by-experiment or
by-session parallelization). Thus, with |cf|_ representing the
computational core of an (arbitrarily complex) superseding algorithm, it
has to be structured to support trial-based parallel computing.
Specifically, this means the scope of work of a |cf|_ is **a single
trial**. Note that this also implies that any parallelism integrated in
|cf|_ has to be designed with higher-level parallel execution in mind
(e.g., concurrent processing of sessions on top of trials).

Technically, a |cf|_ is a regular stand-alone Python function (**not** a
class method) that accepts a :class:`numpy.ndarray` as its first positional
argument and supports (at least) the two keyword arguments `chunkShape` and
`noCompute`. The :class:`numpy.ndarray` represents aggregate data from one
trial (only data, no meta-information). Any required meta-info (such as
channel labels, trial definition records etc.) has to be passed to |cf|_
either as additional (2nd an onward) positional or named keyword arguments
(`chunkShape` and `noCompute` are the only reserved keywords).

The return values of |cf|_ are controlled by the `noCompute` keyword.  In
general, |cf|_ returns exactly one :class:`numpy.ndarray` representing the
result of processing data from a single trial. The `noCompute` keyword is
used to perform a 'dry-run' of the processing operations to propagate the
expected numerical type and memory footprint of the result to
:class:`ComputationalRoutine` without actually performing any
calculations. To optimize performance, :class:`ComputationalRoutine` uses
the information gathered in the dry-runs for each trial to allocate
identically-sized array-blocks accommodating the largest (by shape)
result-array across all trials.  In this manner a global block-size is
identified, which can subsequently be accessed inside |cf|_ via the
`chunkShape` keyword during the actual computation.

Summarized, a valid |cf|_, `cfunc`, meets the following basic requirements:

* **Call signature**

  >>> def cfunc(arr, arg1, arg2, ..., argN, chunkShape=None, noCompute=None, **kwargs)

  where `arr` is a :class:`numpy.ndarray` representing trial data, `arg1`,
  ..., `argN` are arbitrary positional arguments and `chunkShape` (a tuple
  if not `None`) as well as `noCompute` (bool if not `None`) are reserved
  keywords.

* **Return values**

  During the dry-run phase, i.e., if `noCompute` is `True`, the expected
  output shape and its :class:`numpy.dtype` are returned, otherwise the
  result of the computation (a :class:`numpy.ndarray`) is returned:

  >>> def cfunc(arr, arg1, arg2, ..., argN, chunkShape=None, noCompute=None, **kwargs)
  >>> # determine expected output shape and numerical type...
  >>> if noCompute:
  >>>     return outShape, outdtype
  >>> # the actual computation is happening here...
  >>> return res

  Note that dtype and shape of `res` have to agree with `outShape` and
  `outdtype` specified in the dry-run. 

A simple instance of a |cf|_ illustrating these concepts 
is given in :ref:`Examples`.

The Algorithmic Layout of :class:`ComputationalRoutine`
-------------------------------------------------------
Technically, Syncopy's :class:`ComputationalRoutine` wraps an external
|cf|_ by executing all necessary auxiliary routines leading up to and post
termination of the actual computation (memory pre-allocation, generation of
parallel/sequential instruction trees, processing and storage of results,
etc.). Specifically, :class:`ComputationalRoutine` is an abstract base
class that can represent any trial-concurrent computational tree. Thus, any
arbitrarily complex algorithmic pattern satisfying this single criterion
can be incorporated as a regular class into Syncopy with minimal
implementation effort by simply inheriting from
:class:`ComputationalRoutine`.

Internally, the operational principle of a :class:`ComputationalRoutine`
is encapsulated in two class methods:

1. :func:`initialize`

   The class is instantiated with (at least) the positional and keyword
   arguments of the associated |cf|_ minus the trial-data array (the the
   first positional argument of |cf|_) and the reserved keywords
   `chunkShape` and `noCompute`. Further, an additional keyword is reserved
   at class instantiation time: `keeptrials` controls whether data is
   averaged across trials after calculation (``keeptrials = False``).
   Thus, let `Algo` be a concrete subclass of
   :class:`ComputationalRoutine`, and let `cfunc`, defined akin to above

   >>> def cfunc(arr, arg1, arg2, argN, chunkShape=None, noCompute=None, kwarg1="this", kwarg2=False)

   be its corresponding |cf|_. Then a valid instantiation of `Algo` may look
   as follows:

   >>> algorithm = Algo(arg1, arg1, arg2, argN, kwarg1="this", kwarg2=False)

   Now `algorithm` is a regular Python class instance that inherits all
   required attributes from the parent base class
   :class:`ComputationalRoutine`.  **Note**: :class:`ComputationalRoutine`
   uses regular Python class attributes (``__dict__`` keys, not slots) to
   ensure maximal design flexibility for implementing novel computational
   strategies while keeping memory overhead limited due to the
   encapsulation of the actual computational workload in the static method
   |cf|_.

   Before the `algorithm` instance of `Algo` can be used, a dry-run of the
   actual computation has to be performed to determine the expected
   dimensionality and numerical type of the result,

   >>> algorithm.initialize(data)

   where `data` is a Syncopy data object representing the input quantity
   to be processed by `algorithm`.

2. :func:`compute`

   This management method constitutes the functional core of
   :class:`ComputationalRoutine`.  It handles memory pre-allocation,
   storage provisioning, the actual computation and processing of
   meta-information. Theses tasks are encapsulated in distinct class
   methods which are designed to perform the respective operations
   independently from the concrete computational procedure.  Thus, most of
   these methods do not require any problem-specific adaptions and act as
   stand-alone administration routines. The only exception to this
   design-concept is :meth:`process_metadata`, which is intended to attach
   meta-information to the final output object. Since modifications of
   meta-data are highly dependent on the nature of the performed
   calculation, :meth:`process_metadata` is the only abstract method of
   :class:`ComputationalRoutine` that needs to be supplied in addition to
   |cf|_.

   Several keywords control the workflow in :class:`ComputationalRoutine`:

   * Depending on the `parallel` keyword, processing is done either
     sequentially trial by trial (``parallel = False``) or concurrently
     across all trials (if `parallel` is `True`). The two scenarios are
     handled by separate class methods, :func:`compute_sequential` and
     :func:`compute_parallel`, respectively, that use independent
     operational frameworks for processing. However, both
     :func:`compute_sequential` and :func:`compute_parallel` call an
     external |cf|_ to perform the actual calculation.

   * The `parallel_store` keyword controls the employed storage mechanism:
     if `True`, the result of the computation is written in a fully
     concurrent manner where each worker saves its locally held data
     segment on disk leveraging the distributed access capabilities of
     virtual HDF5 datasets. If ``parallel_store = False``, and `parallel` is
     `True`, a mutex is used to lock a single HDF5 file for sequential writing.
     If ``parallel = parallel_store`` and `parallel` is `False`, the computation
     result is saved using standard single-process HDF writing. 

   * The `method` keyword can be used to override the default selection of
     the processing function (:func:`compute_parallel` if `parallel` is
     `True` or :func:`compute_sequential` otherwise). Refer to the
     docstrings of :func:`compute_parallel` or :func:`compute_sequential`
     for details on the required structure of a concurrent or serial
     processing function.

   * The keyword `log_dict` can be used to provide a dictionary of
     keyword-value pairs that are passed on to :meth:`process_metadata` to
     be attached to the final output object.

   Going back to the exemplary `algorithm` instance of `Algo` discussed
   above, after initialization, the actual computation is kicked off with a
   single call of :func:`compute` with keywords pursuant to the intended
   computational workflow. For instance,

   >>> algorithm.compute(data, out, parallel=True)

   launches the parallel processing of `data` using the computational
   scheme implemented in `cfunc` and stores the result in the Syncopy
   object `out`.

To further clarify these concepts, :ref:`Examples` illustrates how to
encapsulate a simple algorithmic scheme in a subclass of
:class:`ComputationalRoutine` that calls a custom |cf|_.

.. _Examples:
       
Examples
--------
Consider the following example illustrating the implementation of a
(deliberately simple) filtering routine by subclassing
:class:`ComputationalRoutine` and designing a |cf|_.  

As a first step, a |cf|_ is defined:

>>> import numpy as np
>>> from scipy import signal
>>> def lowpass(arr, b, a, noCompute=None, chunkShape=None):
>>>     if noCompute:
>>>         return arr.shape, arr.dtype
>>>     res = signal.filtfilt(b, a, arr.T, padlen=200).T
>>>     return res

As detailed above, the first positional argument of `lowpass` is a
:class:`numpy.ndarray` representing numerical data from a single trial, the
second and third positional arguments, `b` and `a` respectively, represent
filter coefficients.  The only keyword arguments of `lowpass` are the
mandatory reserved keywords `noCompute` and `chunkShape`.  With the |cf|_
in place, a subclass of :class:`ComputationalRoutine` can be implemented:

>>> from syncopy.shared.computational_routine import ComputationalRoutine
>>> class LowPassFilter(ComputationalRoutine):
>>>     computeFunction = staticmethod(lowpass)
>>>
>>>     def process_metadata(self, data, out):
>>>         if self.keeptrials:
>>>             out.sampleinfo = np.array(data.sampleinfo)
>>>             out.trialinfo = np.array(data.trialinfo)
>>>             out._t0 = np.zeros((len(data.trials),))
>>>         else:
>>>             trl = np.array([[0, data.sampleinfo[0, 1], 0]])
>>>             out.sampleinfo = trl[:, :2]
>>>             out._t0 = trl[:, 2]
>>>             out.trialinfo = trl[:, 3:]
>>>         out.samplerate = data.samplerate
>>>         out.channel = np.array(data.channel)

Note that `LowPassFilter` simply binds the |cf|_ `lowpass` as static
method - no additional modifications are required. It further provides
`process_metadata` as regular class method for setting all required
attributes of the output object `out`.

Suppose `data` is a Syncopy :class:`AnalogData` object holding data to be
filtered.  To use the introduced filtering routine, the concrete class
`LowPassFilter` has to be instantiated first:

>>> myfilter = LowPassFilter(b, a)

This step performs the actual class initialization and allocates the
attributes of :class:`ComputationalRoutine`. Next, all necessary
pre-calculation management tasks need to be performed:

>>> myfilter.initialize(data)

Now, the `myfilter` instance holds references to the expected shape of the
resulting output and its numerical type. The actual filtering is then
performed by first allocating an empty Syncopy object for the result

>>> out = spy.AnalogData()

and subsequently invoking

>>> myfilter.compute(data, out)

This call performs several tasks: first, an HDF5 data-set of appropriate
dimensions is allocated, then the actual filtering is performed
sequentially, in a trial-by-trial succession (results are stored in the
created HDF5 data-set, which is subsequently attached to `out`), and
finally meta-data is written to the output object `out` using the supplied
`process_metadata` class method.

To perform the calculation in a trial-concurrent manner, first launch a
dask client (using e.g., :func:`syncopy.esi_cluster_setup`), re-initialize
the `myfilter` instance (to reset its attributes) and simply call `compute`
with the `parallel` keyword set to `True`:

>>> client = spy.esi_cluster_setup()
>>> myfilter.initialize(data)
>>> myfilter.compute(data, out, parallel=True)

For realizing more complex mechanisms, consult the implementations of 
:func:`syncopy.freqanalysis` or other metafunctions in Syncopy.
