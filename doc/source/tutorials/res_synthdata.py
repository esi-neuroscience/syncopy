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


# compute the trial averaged spectrum and plot
spec = spy.freqanalysis(adata, keeptrials=False)
# spec.singlepanelplot(channel=0)

# naive downsampling
ds_adata = spy.resampledata(adata, method='downsample', resamplefs=1000)
ds_spec = spy.freqanalysis(ds_adata, keeptrials=False)
f, ax = ds_spec.singlepanelplot(channel=0)
# ax.annotate('?', (315, -1), fontsize=25)


ds_adata2 = spy.resampledata(adata, method='downsample', resamplefs=1000, lpfreq=500)
ds_spec2 = spy.freqanalysis(ds_adata2, keeptrials=False)
f, ax = ds_spec2.singlepanelplot(channel=0)

