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
    syncopy.shared.errors.SPYWarning
    syncopy.shared.kwarg_decorators.unwrap_cfg
    syncopy.shared.kwarg_decorators.unwrap_select
    syncopy.shared.kwarg_decorators.unwrap_io
    syncopy.shared.kwarg_decorators.detect_parallel_client
    syncopy.shared.kwarg_decorators._append_docstring
    syncopy.shared.kwarg_decorators._append_signature
    syncopy.shared.tools.best_match


syncopy.specest
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _stubs

    syncopy.specest.mtmfft.mtmfft
    syncopy.specest.compRoutines.MultiTaperFFT
    syncopy.specest.compRoutines.mtmfft_cF
    syncopy.specest.mtmconvol.mtmconvol
    syncopy.specest.compRoutines.MultiTaperFFTConvol
    syncopy.specest.compRoutines.mtmconvol_cF
    syncopy.specest.compRoutines._make_trialdef
    syncopy.specest.wavelet.wavelet
    syncopy.specest.compRoutines.WaveletTransform
    syncopy.specest.compRoutines.wavelet_cF
    syncopy.specest.compRoutines.SuperletTransform
    syncopy.specest.compRoutines.superlet_cF
    syncopy.specest.compRoutines._make_trialdef
    syncopy.specest.superlet.superlet
    syncopy.specest.wavelet.get_optimal_wavelet_scales

syncopy.plotting
^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: _stubs

    syncopy.plotting.spy_plotting._layout_subplot_panels
    syncopy.plotting.spy_plotting._prep_plots
    syncopy.plotting.spy_plotting._prep_toilim_avg
    syncopy.plotting.spy_plotting._setup_figure
    syncopy.plotting.spy_plotting._setup_colorbar
    syncopy.plotting._plot_spectral._compute_pltArr
    syncopy.plotting._plot_spectral._prep_spectral_plots
    syncopy.plotting._plot_analog._prep_analog_plots
