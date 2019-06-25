function write_spy(filename, data, trialdata, ...
    log, samplerate, version, channel, dimord)

[path, base, ext] = fileparts(filename);

randomHash = spy.hash.DataHash(now, 'SHA-1');
randomHash = randomHash(1:4);

%% HDF5 data file
hdfFile = fullfile(path, [base, '.', randomHash, '.dat']);

% dataset
dclass = 'AnalogData';

dataSize = size(data);
dataSize = dataSize(end:-1:1);

h5create(hdfFile, ['/' dclass], dataSize)
h5write(hdfFile, ['/' dclass], permute(data, ndims(data):-1:1))

trlSize = size(trialdata);
trlSize = trlSize(end:-1:1);
h5create(hdfFile, '/trialdefinition', trlSize)
h5write(hdfFile, '/trialdefinition', trialdata')

% attributes
h5writeatt(hdfFile, '/', 'log', log)
h5writeatt(hdfFile, '/', 'type', dclass)
h5writeatt(hdfFile, '/', 'samplerate', samplerate)
h5writeatt(hdfFile, '/', 'version', version)

% cell arrays don't work with hdf5write
cellstr_h5writeatt(hdfFile, 'channel', channel)
cellstr_h5writeatt(hdfFile, 'dimord', dimord)

%% json info file
hdfHash = spy.hash.DataHash(hdfFile, 'SHA-1', 'file');

jsonFile = fullfile(path, [base, '.', randomHash, '.info']);
spyInfo = spy.SyncopyInfo();


spyInfo.log = log;
spyInfo.version = version;
spyInfo.data = filename;
spyInfo.dimord = dimord;
spyInfo.samplerate = samplerate;
spyInfo.type = dclass;
spyInfo.data_checksum = hdfHash;
% FIXME: h5write always writes double precision
% spyInfo.data_dtype = spy.matlab_to_numpy_dtype(data);
spyInfo.data_dtype = 'float64';
spyInfo.data_shape = size(data);
spyInfo.write_to_file(jsonFile)



function cellstr_h5writeatt(filename, attname, value)

fid = H5F.open(filename, 'H5F_ACC_RDWR', 'H5P_DEFAULT');

% Set variable length string type
VLstr_type = H5T.copy('H5T_C_S1');
H5T.set_size(VLstr_type,'H5T_VARIABLE');

% Create a dataspace for cellstr
H5S_UNLIMITED = H5ML.get_constant_value('H5S_UNLIMITED');
dspace = H5S.create_simple(1,numel(value), H5S_UNLIMITED);

% Create a dataset plist for chunking
plist = H5P.create('H5P_ATTRIBUTE_CREATE');
% H5P.set_chunk(plist,2); % 2 strings per chunk

% Create attribute
attr = H5A.create(fid, attname, VLstr_type, dspace, plist);

% Write data
% H5A.write(attr,VLstr_type,'H5S_ALL','H5S_ALL','H5P_DEFAULT',str);
H5A.write(attr, VLstr_type, value);

% Close file & resources
H5P.close(plist);
H5T.close(VLstr_type);
H5S.close(dspace);
H5A.close(attr);
H5F.close(fid);
