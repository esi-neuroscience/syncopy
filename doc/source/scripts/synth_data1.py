import numpy as np
import syncopy as spy


def generate_noisy_harmonics(nSamples, nChannels, samplerate):

    f1, f2 = 20, 50 # the harmonic frequencies in Hz

    # the sampling times vector
    tvec = np.arange(nSamples) * 1 / samplerate

    # define the two harmonics
    ch1 = np.cos(2 * np.pi * f1 * tvec)
    ch2 = np.cos(2 * np.pi * f2 * tvec)

    # concatenate channels to to trial array
    trial = np.column_stack([ch1, ch2])

    # add some white noise
    trial += 0.5 * np.random.randn(nSamples, nChannels)

    return trial


nTrials = 50
nSamples = 1000
nChannels = 2
samplerate = 500   # in Hz

# collect trials
trials = []
for _ in range(nTrials):
    trial = generate_noisy_harmonics(nSamples, nChannels, samplerate)
    trials.append(trial)

synth_data = spy.AnalogData(trials, samplerate=samplerate)
