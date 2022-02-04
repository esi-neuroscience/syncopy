import numpy as np
import syncopy as spy


nTrials = 50
nSamples = 1000 
nChannels = 2
samplerate = 500   # in Hz

# the sampling times vector needed for construction
tvec = np.arange(nSamples) * 1 / samplerate
# the 30Hz harmonic
harm = np.cos(2 * np.pi * 30 * tvec)
# the damped amplitudes
dampening = np.linspace(1, 0.1, nSamples)
signal = dampening * harm

# collect trials
trials = []
for _ in range(nTrials):

    white_noise = np.random.randn(nSamples, nChannels)    
    trial = np.tile(signal, (2, 1)).T + white_noise
    trials.append(trial)

# instantiate Syncopy data object
synth_data = spy.AnalogData(trials, samplerate=samplerate)
