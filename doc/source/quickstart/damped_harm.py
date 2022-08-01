import numpy as np
import syncopy as spy


nTrials = 50
nSamples = 1000
nChannels = 2
samplerate = 500   # in Hz

# the sampling times vector needed for construction
tvec = np.arange(nSamples) * 1 / samplerate
# the 30Hz harmonic
harm30 = np.cos(2 * np.pi * 30 * tvec)
# linear dampening down to 10% of the original amplitude
dampening = np.linspace(1, 0.1, nSamples)
signal = dampening * harm30

# collect trials
trials = []
for _ in range(nTrials):

    # we start with the white noise
    trial = np.random.randn(nSamples, nChannels)
    # now add the damped harmonic on the 1st channel
    trial[:, 0] += signal

    trials.append(trial)

# instantiate Syncopy data object
data = spy.AnalogData(trials, samplerate=samplerate)
