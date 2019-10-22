API for Developers
------------------

syncopy.datatype
^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _stubs
    :template: syncopy_class.rst

    syncopy.datatype.base_data.BaseData
    syncopy.datatype.base_data.Selector
    syncopy.datatype.base_data.FauxTrial
    syncopy.datatype.base_data.StructDict
    syncopy.datatype.continuous_data.ContinuousData
    syncopy.datatype.discrete_data.DiscreteData


syncopy.misc
^^^^^^^^^^^^

.. autosummary::
    :toctree: _stubs

    syncopy.tests.misc.generate_artificial_data


syncopy.shared
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _stubs

    syncopy.shared.computational_routine.ComputationalRoutine
    syncopy.shared.errors.SPYError
    syncopy.shared.errors.SPYTypeError
    syncopy.shared.errors.SPYValueError
    syncopy.shared.errors.SPYIOError
    syncopy.shared.kwarg_decorators.unwrap_cfg
    syncopy.shared.kwarg_decorators.unwrap_select
    syncopy.shared.kwarg_decorators.unwrap_io
    syncopy.shared.kwarg_decorators._append_docstring
    syncopy.shared.kwarg_decorators._append_signature


syncopy.specest
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _stubs

    syncopy.specest.mtmfft.mtmfft
    syncopy.specest.mtmfft.MultiTaperFFT
    syncopy.specest.wavelet.wavelet
    syncopy.specest.wavelet.WaveletTransform
