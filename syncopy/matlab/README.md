MATLAB interface for Syncopy files
==================================

The functions provided here offer loading and saving routines for Syncopy
data files, i.e., data arrays stored in an HDF5 file with metadata in a JSON
file. 

To get started, simply add the folder containing the `+spy` directory to your 
MATLAB path. It is not necessary to include all subfolders. 

For both loading and saving there are two different routines. The "low-level"
functions work with data arrays (matrices):
* `spy.save_spy`
* `spy.load_spy`

As Syncopy aims to be compatible with the MATLAB toolbox Fieldtrip [1], there
are also "high-level" functions that operate with Fieldtrip data structs:
* `spy.ft_save_spy`
* `spy.ft_load_spy`

As a shortcut, you can issue an `import` command at the beginning of your code:
```matlab
import spy.*
ft_save_spy(filename)
```

The Syncopy-MATLAB interface uses and distributes external code according to
their licenses (JSONlab [2], DataHash [3]). This code is distributed under the
terms of the BSD-3-Clause license. See the file LICENSE for more details.

References
----------
[1] https://www.fieldtriptoolbox.org

[2] Qianqian Fang (2019). JSONlab: a toolbox to encode/decode JSON files 
    (https://www.mathworks.com/matlabcentral/fileexchange/33381-jsonlab-a-toolbox-to-encode-decode-json-files),
    MATLAB Central File Exchange. Retrieved July 4, 2019. 

[3] Jan (2019). DataHash 
    (https://www.mathworks.com/matlabcentral/fileexchange/31272-datahash), 
    MATLAB Central File Exchange. Retrieved July 4, 2019. 