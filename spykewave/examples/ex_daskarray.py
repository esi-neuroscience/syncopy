# ex_daskarray.py - Script for testing Dask arrays vs. bags
# 
# Created: Januar 25 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-01-29 12:08:32>

# Builtin/3rd party package imports
import dask
import dask.array as da
import dask.bag as db
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from dask.diagnostics import ProgressBar
import numpy as np
import scipy.signal.windows as windows
import matplotlib.pyplot as plt
import os
import sys
import time
import timeit

# Add spykewave package to Python search path
spw_path = os.path.abspath(".." + os.sep + "..")
if spw_path not in sys.path:
    sys.path.insert(0, spw_path)

# Import Spykewave
import spykewave as sw

# Binary flags controlling program flow
slurmComputation = False        # turn SLURM usage on/off and
demo = True                     # compute single-trial powerspectra and plot results
benchmark = False               # benchmark parallelization techniques

# %% #################### Demo analysis ####################
if demo:

    # %% -------------------- Set up parallel environment --------------------
    if slurmComputation:
        cluster = SLURMCluster(processes=8,
                               cores=8,
                               memory="48GB",                                              
                               queue="DEV")
        cluster.start_workers(1)    
        print("Waiting for workers to start")    
        while len(cluster.workers) == 0:
            time.sleep(0.5)
        client = Client(cluster)
        print(client)


    # %% -------------------- Define location of test data --------------------
    # datadir = "/mnt/hpx/it/dev/SpykeWave/testdata/"
    datadir = ".." + os.sep + ".." + os.sep + ".." + os.sep + ".." + os.sep + "Data"\
              + os.sep + "testdata" + os.sep
    basename = "MT_RFmapping_session-168a1"


    # %% -------------------- Define trial from photodiode onsets --------------------
    pdFile = os.path.join(datadir, basename + ".dpd")
    pdData = sw.BaseData(pdFile, filetype="esi")

    # Helper functions to convert sample-no to actual time-code
    def time2sample(t, dt=0.001):
        return (t/dt).astype(dtype=int)
    def sample2time(s, dt=0.001):
        return s*dt

    # Trials start 250 ms before stimulus onset
    pdOnset = np.array(pdData._segments[:,:])
    iStart = time2sample(sample2time(pdOnset[0, pdOnset[1,:] == 1], 
                                     dt=pdData.hdr["tSample"]/1E9) - 0.25,
                         dt=0.001)
    iEnd = time2sample(sample2time(pdOnset[0, pdOnset[1,:] == 1],
                                   dt=pdData.hdr["tSample"]/1E9) + 0.5,
                       dt=0.001)

    # Construct trial definition matrix
    intervals = np.stack((iStart,iEnd, np.tile(250, [iStart.size]).T), axis=1)

    # Remove very short trials
    intervals = intervals[intervals[:,1]-intervals[:,0] > 500]

    # Read actual raw data from disk
    dataFiles = [os.path.join(datadir, basename + ext) for ext in ["_xWav.lfp", "_xWav.mua"]]
    data = sw.BaseData(dataFiles, trialdefinition=intervals, filetype="esi")


    # %% --------------------Prepare spectral analysis --------------------
    cfg = sw.spw_get_defaults(sw.mtmfft)
    cfg.pop("data")
    cfg["taper"] = windows.dpss
    cfg["fftAxis"] = 1
    cfg["pad"] = "nextpow2"
    cfg["tapsmofrq"] = 5

    # Point to data segments on disk by using delayed method calls
    get_segment = dask.delayed(data.get_segment)
    lazy_values = [get_segment(trialno) for trialno in range(len(data.trialinfo))]

    # Construct a distributed dask array by stacking delayed segments of appropriate (identical) shapes
    stack = da.stack([da.from_delayed(lazy_value,
                                      shape=(512, 750),
                                      dtype=data.hdr['dtype'])
                      for lazy_value in lazy_values])

    # Construct a distributed dask bag from a sequence of arbitrary(!) objects
    # Note: this construction:
    #       bag = db.from_delayed([lazy_value for lazy_value in lazy_values])
    # loads segments row-wise (for whatever reason) while
    #       bag = db.from_sequence([lazy_value for lazy_value in lazy_values])
    # does not work since the delayed lazy values are not resolved correctly when
    # mapping to mtmfft ("Truth of Delayed objects is not supported")
    bag = db.from_sequence(range(len(data._sampleinfo)))

    # Same as above but only use the first two segments
    small_bag = db.from_sequence(range(2))


    # %% -------------------- Perform spectral analysis --------------------
    # Prototype function that returns a single array
    def mtmfft_single_out_wrapper(seg):
        freq, spec, win = sw.mtmfft(seg.squeeze(), **cfg)
        spec = np.mean(np.absolute(spec), axis=0)
        return spec[np.newaxis, :, :]

    # Prototype function tailored for range-bags
    def mtmfft_bag_wrapper(trialno, data):
        freq, spec, win = sw.mtmfft(data.get_segment(trialno), **cfg)
        spec = np.mean(np.absolute(spec), axis=0)
        return spec

    # Prototype function that returns a tuple of arrays (could be done w/ a gufunc as well)
    def mtmfft_multi_out_wrapper(trialno, data):
        freq, spec, win = sw.mtmfft(data.get_segment(trialno), **cfg)
        spec = np.mean(np.absolute(spec), axis=0)
        return freq, spec, win

    # Use `map_blocks` to compute spectra for each segment in the constructred dask array
    specs_stack = stack.map_blocks(mtmfft_single_out_wrapper, chunks=(1, 512, 513))

    # Use `map` to compute spectra for each element in the constructred dask bag
    specs_bag = bag.map(mtmfft_bag_wrapper, data=data)

    # Same as above, but keep multiple outputs per method call (hence we only do this for the first two segments)
    freq_spec_win = small_bag.map(mtmfft_multi_out_wrapper, data=data)

    # Compute spectra w/ the constructed dask array/bags
    print("Computing single trial powerspectra using a dask array")
    with ProgressBar():
        result_stack = specs_stack.compute()        # no stacking necessary, result is already a dask array
    print("Computing single trial powerspectra using a dask bag")
    with ProgressBar():
        result_bag = np.stack(specs_bag.compute())
    print("Computing single trial powerspectra using a smaller dask bag w/ multiple outputs")
    with ProgressBar():
        result_sb = freq_spec_win.compute()
    result_small_bag = np.stack([res[1] for res in result_sb])


    # %% -------------------- Plot results --------------------
    # Compute ERP, construct time-axis and get frequencies
    erp = np.zeros(data.get_segment(0).shape) 
    for segment in data.segments:
        erp += segment
    erp /= len(data._trialinfo)
    tAxis = np.arange(erp.shape[1]) * data.hdr["tSample"]/1E6 - 250
    channels = np.arange(0, 15)
    freq, _, _ = sw.mtmfft(data.get_segment(0), **cfg)

    # Turn on interactive plotting and render one figure per computed powerspectrum
    plt.ion()
    titles = ["Dask Array", "Dask Bag", "Small Bag"]
    for rk, result in enumerate([result_stack, result_bag, result_small_bag]):
        avgSpec = result.mean(axis=0)
        fig, ax = plt.subplots(2,2, sharex="row")
        # fig, ax = plt.subplots(2,2, tight_layout=True, sharex="row")
        fig.suptitle(titles[rk])
        ax[0,0].plot(tAxis, erp[channels,:].T)
        ax[0,1].plot(tAxis, erp[channels+256,:].T)
        ax[0,0].set_ylabel('LFP (a.u.)')
        ax[0,1].set_ylabel('MUA (a.u.)')
        ax[0,0].set_xlabel('Time (ms)')
        ax[1,0].semilogy(freq, avgSpec[channels,:].T)
        ax[1,0].set_xlim([1, 100])
        ax[1,1].semilogy(freq, avgSpec[channels+256,:].T)
        ax[1,0].set_xlabel('Frequency (Hz)')
        ax[1,0].set_ylabel('Power (a.u.)')
    plt.draw()


# %% #################### Time computations ####################
if benchmark:

    # Set number of repitiions and no. of loops to run + output message template
    reps = 3
    nmbr = int(10)
    msg = "{nmbr:d} loops, best of {reps:d}: {t:3.2f} sec per loop"

    # The setup part common to all profiling runs: setup code is executed only once and not timed
    setup = """
# Builtin/3rd party package imports
import dask
import dask.array as da
import dask.bag as db
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from dask.diagnostics import ProgressBar
import numpy as np
import scipy.signal.windows as windows
import matplotlib.pyplot as plt
import os
import sys
import time

# Add spykewave package to Python search path
spw_path = os.path.abspath(".." + os.sep + "..")
if spw_path not in sys.path:
    sys.path.insert(0, spw_path)

# Import Spykewave
import spykewave as sw

# Binary flags controlling program flow
slurmComputation = False        # turn SLURM usage on/off and

# %% -------------------- Set up parallel environment --------------------
if slurmComputation:
    cluster = SLURMCluster(processes=8,
                           cores=8,
                           memory="48GB",                                              
                           queue="DEV")
    cluster.start_workers(1)    
    print("Waiting for workers to start")    
    while len(cluster.workers) == 0:
        time.sleep(0.5)
    client = Client(cluster)
    print(client)


# %% -------------------- Define location of test data --------------------
# datadir = "/mnt/hpx/it/dev/SpykeWave/testdata/"
datadir = ".." + os.sep + ".." + os.sep + ".." + os.sep + ".." + os.sep + "Data"\
          + os.sep + "testdata" + os.sep
basename = "MT_RFmapping_session-168a1"


# %% -------------------- Define trial from photodiode onsets --------------------
pdFile = os.path.join(datadir, basename + ".dpd")
pdData = sw.BaseData(pdFile, filetype="esi")

# Helper functions to convert sample-no to actual time-code
def time2sample(t, dt=0.001):
    return (t/dt).astype(dtype=int)
def sample2time(s, dt=0.001):
    return s*dt

# Trials start 250 ms before stimulus onset
pdOnset = np.array(pdData._segments[:,:])
iStart = time2sample(sample2time(pdOnset[0, pdOnset[1,:] == 1], 
                                 dt=pdData.hdr["tSample"]/1E9) - 0.25,
                     dt=0.001)
iEnd = time2sample(sample2time(pdOnset[0, pdOnset[1,:] == 1],
                               dt=pdData.hdr["tSample"]/1E9) + 0.5,
                   dt=0.001)

# Construct trial definition matrix
intervals = np.stack((iStart,iEnd, np.tile(250, [iStart.size]).T), axis=1)

# Remove very short trials
intervals = intervals[intervals[:,1]-intervals[:,0] > 500]

# Read actual raw data from disk
dataFiles = [os.path.join(datadir, basename + ext) for ext in ["_xWav.lfp", "_xWav.mua"]]
data = sw.BaseData(dataFiles, trialdefinition=intervals, filetype="esi")


# %% --------------------Prepare spectral analysis --------------------
cfg = sw.spw_get_defaults(sw.mtmfft)
cfg.pop("data")
# print(cfg)
cfg["taper"] = windows.dpss
cfg["fftAxis"] = 1
cfg["pad"] = "nextpow2"
cfg["tapsmofrq"] = 5

# Point to data segments on disk by using delayed method calls
get_segment = dask.delayed(data.get_segment)
lazy_values = [get_segment(trialno) for trialno in range(len(data.trialinfo))]

    """
    
    # Time dask array
    setup_stack = setup + """
# Construct a distributed dask array by stacking delayed segments of appropriate (identical) shapes
stack = da.stack([da.from_delayed(lazy_value,
                                  shape=(512, 750),
                                  dtype=data.hdr['dtype'])
                  for lazy_value in lazy_values])

# Prototype function that returns a single array
def mtmfft_single_out_wrapper(seg):
    freq, spec, win = sw.mtmfft(seg.squeeze(), **cfg)
    spec = np.mean(np.absolute(spec), axis=0)
    return spec[np.newaxis, :, :]

# Use `map_blocks` to compute spectra for each segment in the constructred dask array
specs_stack = stack.map_blocks(mtmfft_single_out_wrapper, chunks=(1, 512, 513))

    """
    timings = timeit.repeat(setup=setup_stack,
                            stmt="specs_stack.compute()",
                            repeat=reps, number=nmbr)
    print("Dask array: " + msg.format(nmbr=nmbr, reps=reps, t=min(timings)))

    # Time dask bag
    setup_bag = setup + """
# Construct a distributed dask bag from a sequence of arbitrary(!) objects
bag = db.from_sequence(range(len(data._sampleinfo)))

# Prototype function tailored for range-bags
def mtmfft_bag_wrapper(trialno, data):
    freq, spec, win = sw.mtmfft(data.get_segment(trialno), **cfg)
    spec = np.mean(np.absolute(spec), axis=0)
    return spec

# Use `map` to compute spectra for each element in the constructred dask bag
specs_bag = bag.map(mtmfft_bag_wrapper, data=data)

    """
    timings = timeit.repeat(setup=setup_bag,
                            stmt="specs_bag.compute()",
                            repeat=reps, number=nmbr)
    print("Dask bag: " + msg.format(nmbr=nmbr, reps=reps, t=min(timings)))
    
    # Time small dask bag using multiple outputs
    setup_small = setup + """
# Same as above but only use the first two segments
small_bag = db.from_sequence(range(2))

# Prototype function that returns a tuple of arrays (could be done w/ a gufunc as well)
def mtmfft_multi_out_wrapper(trialno, data):
    freq, spec, win = sw.mtmfft(data.get_segment(trialno), **cfg)
    spec = np.mean(np.absolute(spec), axis=0)
    return freq, spec, win

# Same as above, but keep multiple outputs per method call (hence we only do this for the first two segments)
freq_spec_win = small_bag.map(mtmfft_multi_out_wrapper, data=data)

    """
    timings = timeit.repeat(setup=setup_small,
                            stmt="freq_spec_win.compute()",
                            repeat=reps, number=nmbr)
    print("Small dask bag: " + msg.format(nmbr=nmbr, reps=reps, t=min(timings)))
