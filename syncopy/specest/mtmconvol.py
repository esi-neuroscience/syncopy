# -*- coding: utf-8 -*-
# 
# Time-frequency analysis based on a short-time Fourier transform
# 
# Created: 2020-02-05 09:36:38
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-02-07 16:04:32>

from syncopy.shared.kwarg_decorators import unwrap_io


# Local workhorse that performs the computational heavy lifting
@unwrap_io
def mtmconvol(trl_dat, dt, nTaper=1, timeAxis=0,
              taper=spwin.hann, taperopt={}, tapsmofrq=None,
              pad="nextpow2", padtype="zero", padlength=None, 
              prepadlength=True, postpadlength=False,
              foi=None,
              keeptapers=True, polyorder=None, output_fmt="pow",
              noCompute=False, chunkShape=None):
    """
    Coming soon...
    """
    
    # Re-arrange array if necessary and get dimensional information
    if timeAxis != 0:
        dat = trl_dat.T       # does not copy but creates view of `trl_dat`
    else:
        dat = trl_dat

    # Padding (updates no. of samples)
    if pad is not None:
        padKw = padding(dat, padtype, pad=pad, padlength=padlength, 
                        prepadlength=prepadlength, postpadlength=postpadlength,
                        create_new=False)
        padbegin = max(0, padbegin - padKw["pad_width"][0, 0])
        padend = max(0, padend - padKw["pad_width"][0, 1])
        dat = np.pad(dat, **padKw)
    nSamples = dat.shape[0]
    nChannels = dat.shape[1]
