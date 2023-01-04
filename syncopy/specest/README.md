# specest - Power Spectral Estimation

## User Frontend

- [freqanalysis.py](./freqanalysis.py): parent metafunction to access all implemented spectral estimation methods

## Available Methods

- [mtmfft](./mtmfft.py): (multi-)tapered windowed Fourier transform, returns a periodogram estimate
- [mtmconvol](./mtmconvol.py): (multi-)tapered windowed Fourier transform, returns time-frequency representation
- [wavelet](./wavelet.py): Wavelet analysis based on [Torrence and Compo, 1998](https://cobblab.eas.gatech.edu/seminar/torrence&compo98.pdf)
- [superlet](./superlet.py): Superlet transform as proposed in [Moca et al. 2021](https://www.nature.com/articles/s41467-020-20539-9) (coming soon..)

## Usage Examples (TODO..)

...

## Sources

- [Wavelet core library](./wavelets/) from GitHub: https://github.com/aaren/wavelets
