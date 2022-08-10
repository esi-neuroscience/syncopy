# Import package
import syncopy as spy
from syncopy.tests import synth_data

# basic dataset properties
nTrials = 100
samplerate = 5000  # in Hz

# add a harmonic with 200Hz
adata = 2 * synth_data.harmonic(nTrials, freq=200, samplerate=samplerate)

# add another harmonic with 1300Hz
adata += synth_data.harmonic(nTrials, freq=1300, samplerate=samplerate)

# white noise floor
adata += synth_data.white_noise(nTrials, samplerate=samplerate)

# compute the trial averaged spectrum and plot
spec = spy.freqanalysis(adata, keeptrials=False)
f1, ax1 = spec.singlepanelplot(channel=0)
f1.set_size_inches(6.4, 3)
f1.savefig('res_orig_spec.png')

# naive downsampling
ds_adata = spy.resampledata(adata, method='downsample', resamplefs=1000)
ds_spec = spy.freqanalysis(ds_adata, keeptrials=False)
f2, ax2 = ds_spec.singlepanelplot(channel=0)
ax2.annotate('?', (315, -.5), fontsize=25)
f2.set_size_inches(6.4, 3)
f2.savefig('res_ds_spec.png')


ds_adata2 = spy.resampledata(adata, method='downsample', resamplefs=1000, lpfreq=500)
ds_spec2 = spy.freqanalysis(ds_adata2, keeptrials=False)
f3, ax3 = ds_spec2.singlepanelplot(channel=0)
f3.set_size_inches(6.4, 3)
f3.savefig('res_lpds_spec.png')

ds_adata3 = spy.resampledata(adata, method='downsample', resamplefs=1000, lpfreq=500, order=5000)
ds_spec3 = spy.freqanalysis(ds_adata3, keeptrials=False)
f4, ax4 = ds_spec3.singlepanelplot(channel=0)
f4.set_size_inches(6.4, 3)
f4.savefig('res_lporderds_spec.png')

# resampling

# rs_adata = spy.resampledata(adata, method='resample', resamplefs=1202, order=20000)
rs_adata = spy.resampledata(adata, method='resample', resamplefs=1200)
rs_spec = spy.freqanalysis(rs_adata, keeptrials=False)
f5, ax5 = rs_spec.singlepanelplot(channel=0)
f5.set_size_inches(6.4, 3)
f5.savefig('res_rs_spec.png')
