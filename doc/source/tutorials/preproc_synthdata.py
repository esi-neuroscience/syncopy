# built-in synthetic data generators
import syncopy as spy
from syncopy import synthdata as spy_synth

cfg_synth = spy.StructDict()
cfg_synth.nTrials = 150
cfg_synth.samplerate = 500
cfg_synth.nSamples = 1000
cfg_synth.nChannels = 2

# 30Hz undamped harmonig
harm = spy_synth.harmonic(cfg_synth, freq=30)

# a linear trend
lin_trend = spy_synth.linear_trend(cfg_synth, y_max=3)

# a 2nd 'nuisance' harmonic
harm50 = spy_synth.harmonic(cfg_synth, freq=50)

# finally the white noise floor
wn = spy_synth.white_noise(cfg_synth)
