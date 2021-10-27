# -*- coding: utf-8 -*-
#
# Helper methods for testing routines
#

# Builtin/3rd party package imports
import subprocess
import sys
import os
import h5py
import tempfile
import time
import numpy as np

# Local imports
from syncopy.datatype import AnalogData
from syncopy.shared.filetypes import _data_classname_to_extension, FILE_EXT
from syncopy import __plt__, __acme__
if __plt__:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
if __acme__:
    import dask.distributed as dd


def is_win_vm():
    """
    Returns `True` if code is running on virtual Windows machine, `False`
    otherwise
    """

    # If we're not running on Windows abort
    if sys.platform != "win32":
        return False

    # Use the windows management instrumentation command-line to extract machine manufacturer
    out, err = subprocess.Popen("wmic computersystem get manufacturer",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, shell=True).communicate()

    # If the vendor name contains any "virtual"-flavor, we're probably running
    # in a VM - if the above command triggered an error, abort
    if len(err) == 0:
        vendor = out.split()[1].lower()
        vmlist = ["vmware", "virtual", "virtualbox", "vbox", "qemu"]
        return any([virtual in vendor for virtual in vmlist])
    else:
        return False


def is_slurm_node():
    """
    Returns `True` if code is running on a SLURM-managed cluster node, `False`
    otherwise
    """

    # Simply test if the srun command is available
    out, err = subprocess.Popen("srun --version",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, shell=True).communicate()
    if len(out) > 0:
        return True
    else:
        return False


def generate_artificial_data(nTrials=2, nChannels=2, equidistant=True, seed=None,
                             overlapping=False, inmemory=True, dimord="default"):
    """
    Create :class:`~syncopy.AnalogData` object with synthetic harmonic signal(s)

    Parameters
    ----------
    nTrials : int
        Number of trials to populate synthetic data object with
    nChannels : int
        Number of channels to populate synthetic object with
    equidistant : bool
        If `True`, trials of equal length are defined
    seed : None or int
        If `None`, imposed noise is completely random. If `seed` is an integer,
        it is used to fix the (initial) state of NumPy's random number generator
        :func:`numpy.random.default_rng`, i.e., objects created wtih same `seed`
        will be populated with identical artificial signals.
    overlapping : bool
        If `True`, constructed trials overlap
    inmemory : bool
        If `True`, the full `data` array (all channels across all trials) is allocated
        in memory (fast but dangerous for large arrays), otherwise the output data
        object's corresponding backing HDF5 file in `__storage__` is filled with
        synthetic data in a trial-by-trial manner (slow but safe even for very
        large datasets).
    dimord : str or list
        If `dimord` is "default", the constructed output object uses the default
        dimensional layout of a standard :class:`~syncopy.AnalogData` object.
        If `dimord` is a list (i.e., ``["channel", "time"]``) the provided sequence
        of dimensions is used.

    Returns
    -------
    out : :class:`~syncopy.AnalogData` object
        Syncopy :class:`~syncopy.AnalogData` object with specified properties
        populated with a synthetic multivariate trigonometric signal.

    Notes
    -----
    This is an auxiliary method that is intended purely for internal use. Thus,
    no error checking is performed.

    Examples
    --------
    Generate small artificial :class:`~syncopy.AnalogData` object in memory

    .. code-block:: python

        >>> iAmSmall = generate_artificial_data(nTrials=5, nChannels=10, inmemory=True)
        >>> iAmSmall
        Syncopy AnalogData object with fields

                    cfg : dictionary with keys ''
                channel : [10] element <class 'numpy.ndarray'>
              container : None
                   data : 5 trials of length 3000 defined on [15000 x 10] float32 Dataset of size 0.57 MB
                 dimord : 2 element list
               filename : /Users/pantaray/.spy/spy_158f_4d4153e3.analog
                   mode : r+
             sampleinfo : [5 x 2] element <class 'numpy.ndarray'>
             samplerate : 1000.0
                    tag : None
                   time : 5 element list
              trialinfo : [5 x 0] element <class 'numpy.ndarray'>
                 trials : 5 element iterable

        Use `.log` to see object history

    Generate artificial :class:`~syncopy.AnalogData` object of more substantial
    size on disk

    .. code-block:: python

        >>> iAmBig = generate_artificial_data(nTrials=50, nChannels=1024, inmemory=False)
        >>> iAmBig
        Syncopy AnalogData object with fields

                    cfg : dictionary with keys ''
                channel : [1024] element <class 'numpy.ndarray'>
              container : None
                   data : 200 trials of length 3000 defined on [600000 x 1024] float32 Dataset of size 2.29 GB
                 dimord : 2 element list
               filename : /Users/pantaray/.spy/spy_158f_b80715fe.analog
                   mode : r+
             sampleinfo : [200 x 2] element <class 'numpy.ndarray'>
             samplerate : 1000.0
                    tag : None
                   time : 200 element list
              trialinfo : [200 x 0] element <class 'numpy.ndarray'>
                 trials : 200 element iterable

        Use `.log` to see object history

    """

    # Create dummy 1d signal that will be blown up to fill channels later
    dt = 0.001
    t = np.arange(0, 3, dt, dtype="float32") - 1.0
    sig = np.cos(2 * np.pi * (7 * (np.heaviside(t, 1) * t - 1) + 10) * t)

    # Depending on chosen `dimord` either get default position of time-axis
    # in `AnalogData` objects or use provided `dimord` and reshape signal accordingly
    if dimord == "default":
        dimord = AnalogData._defaultDimord
    timeAxis = dimord.index("time")
    idx = [1, 1]
    idx[timeAxis] = -1
    sig = np.repeat(sig.reshape(*idx), axis=idx.index(1), repeats=nChannels)

    # Initialize random number generator (with possibly user-provided seed-value)
    rng = np.random.default_rng(seed)

    # Either construct the full data array in memory using tiling or create
    # an HDF5 container in `__storage__` and fill it trial-by-trial
    # NOTE: use `swapaxes` here to ensure two objects created w/same seed really
    # are affected w/identical additive noise patterns, no matter their respective
    # `dimord`.
    out = AnalogData(samplerate=1/dt, dimord=dimord)
    if inmemory:
        idx[timeAxis] = nTrials
        sig = np.tile(sig, idx)
        shp = [slice(None), slice(None)]
        for iTrial in range(nTrials):
            shp[timeAxis] = slice(iTrial*t.size, (iTrial + 1)*t.size)
            noise = rng.standard_normal((t.size, nChannels)).astype(sig.dtype) * 0.5
            sig[tuple(shp)] += np.swapaxes(noise, timeAxis, 0)
        out.data = sig
    else:
        with h5py.File(out.filename, "w") as h5f:
            shp = list(sig.shape)
            shp[timeAxis] *= nTrials
            dset = h5f.create_dataset("data", shape=tuple(shp), dtype=sig.dtype)
            shp = [slice(None), slice(None)]
            for iTrial in range(nTrials):
                shp[timeAxis] = slice(iTrial*t.size, (iTrial + 1)*t.size)
                noise = rng.standard_normal((t.size, nChannels)).astype(sig.dtype) * 0.5
                dset[tuple(shp)] = sig + np.swapaxes(noise, timeAxis, 0)
                dset.flush()
        out.data = h5py.File(out.filename, "r+")["data"]

    # Define by-trial offsets to generate (non-)equidistant/(non-)overlapping trials
    trialdefinition = np.zeros((nTrials, 3), dtype='int')
    if equidistant:
        equiOffset = 0
        if overlapping:
            equiOffset = 100
        offsets = np.full((nTrials,), equiOffset, dtype=sig.dtype)
    else:
        offsets = rng.integers(low=int(0.1*t.size), high=int(0.2*t.size), size=(nTrials,))

    # Using generated offsets, construct trialdef array and make sure initial
    # and end-samples are within data bounds (only relevant if overlapping
    # trials are built)
    shift = (-1)**(not overlapping)
    for iTrial in range(nTrials):
        trialdefinition[iTrial, :] = np.array([iTrial*t.size - shift*offsets[iTrial],
                                               (iTrial + 1)*t.size + shift*offsets[iTrial],
                                               -1000])
    if equidistant:
        trialdefinition[0, :2] += equiOffset
        trialdefinition[-1, :2] -= equiOffset
    else:
        trialdefinition[0, 0] = 0
        trialdefinition[-1, 1] = nTrials*t.size
    out.definetrial(trialdefinition)

    return out


def construct_spy_filename(basepath, obj):
    basename = os.path.split(basepath)[1]
    objext = _data_classname_to_extension(obj.__class__.__name__)
    return os.path.join(basepath + FILE_EXT["dir"], basename + objext)


def figs_equal(fig1, fig2, tol=None):
    """
    Test if two figures are identical

    Parameters
    ----------
    fig1 : matplotlib figure object
        Reference figure
    fig2 : matplotlib figure object
        Template figure
    tol : float
        Positive scalar (b/w 0 and 1) specifying tolerance level for considering
        `fig1` and `fig2` identical. If `None`, two figures have to be exact
        pixel-perfect copies to be qualified as identical.

    Returns
    -------
    equal : bool
        `True` if `fig1` and `fig2` are identical, `False` otherwise

    Notes
    -----
    This is an auxiliary method that is intended purely for internal use. Thus,
    no error checking is performed.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> fig1 = plt.figure(); plt.plot(x, np.sin(x))
    >>> figs_equal(fig1, fig1)
    True
    >>> fig2 = plt.figure(); plt.plot(x, np.sin(x), color="red")
    >>> figs_equal(fig1, fig2)
    False
    >>> figs_equal(fig1, fig2, tol=0.9)
    True
    """
    plt.draw_all(force=True)
    with tempfile.NamedTemporaryFile(suffix='.png') as img1:
        with tempfile.NamedTemporaryFile(suffix='.png') as img2:
            fig1.savefig(img1.name)
            fig2.savefig(img2.name)
            if tol is None:
                return np.array_equal(plt.imread(img1.name), plt.imread(img2.name))
            return np.allclose(plt.imread(img1.name), plt.imread(img2.name), atol=tol)


def flush_local_cluster(testcluster, timeout=10):
    """
    Resets a parallel computing client to avoid memory spilling
    """
    if isinstance(testcluster, dd.LocalCluster):
        # client.restart()
        client = dd.get_client()
        client.close()
        time.sleep(1.0)
        client = dd.Client(testcluster)
        waiting = 0
        while len([w["memory_limit"] for w in testcluster.scheduler_info["workers"].values()]) == 0 \
            and waiting < timeout:
                time.sleep(1.0)
                waiting += 1
    return
