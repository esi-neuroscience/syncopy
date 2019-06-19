function write_spy(filename, data, trialdata, ...
    log, dtype, samplerate, version, channel, dimord)

sz = size(data);
sz = sz(end:-1:1);
dclass = 'AnalogData';
h5create(filename, ['/' dclass], sz)
h5write(filename, ['/' dclass], permute(data, ndims(data):-1:1))

sz = size(trialdata);
sz = sz(end:-1:1);
h5create(filename, '/trialdefinition', sz)
h5write(filename, '/trialdefinition', trialdata')

h5writeatt(filename, '/', 'log', log)
h5writeatt(filename, '/', 'type', dtype)
h5writeatt(filename, '/', 'samplerate', samplerate)
h5writeatt(filename, '/', 'version', version)

% cell arrays don't work with hdf5write
cellstr_h5writeatt(filename, 'channel', channel)
cellstr_h5writeatt(filename, 'dimord', dimord)

% json info file
spyInfo = spy.SyncopyInfo();
spyInfo.data_dtype = dtype;
spyInfo.log = log;
spyInfo.version = version;
spyInfo.data = filename;
spyInfo.dimord = dimord;
spyInfo.samplerate = samplerate
spyInfo.type = dclass;
spyInfo.write_to_file('test.json')



function cellstr_h5writeatt(filename, attname, value)

fid = H5F.open(filename, 'H5F_ACC_RDWR', 'H5P_DEFAULT');

% Set variable length string type
VLstr_type = H5T.copy('H5T_C_S1');
H5T.set_size(VLstr_type,'H5T_VARIABLE');

% Create a dataspace for cellstr
H5S_UNLIMITED = H5ML.get_constant_value('H5S_UNLIMITED');
dspace = H5S.create_simple(1,numel(value),H5S_UNLIMITED);

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
