function [hdfFile, jsonFile, spyInfo] = save_spy(filename, ...
    data, trialdefinition, log, samplerate, channel, dimord, ...
    varargin)
% SPY.WRITE_SPY Write data array to Syncopy-compatible HDF5/JSON files
%
%   write_spy(filename, data, trialdefinition, log, samplerate, channel, dimord, ...
%             {dclass, cfg})
%
% INPUT
% -----
%  filename   : filename to be used for saving. The resulting filenames will 
%               be <filename>.analog (HDF5) and <filename>.analog.info (JSON).
%  data       : data array to be stored in HDF5 file
%  trialdefinition : [nTrials x 3+N] array describing beginning, end and
%                    time zero of each trial as well as N additional
%                    columns with scalar metadata about each trial.
%  log        : char array containing a prosaic log of the data history
%  samplerate : sampling rate of the data in Hz
%  channel    : cell array with channel names
%  dimord     : cell array with array dimension names
%
% OUTPUT
% ------
%   hdfFile   : filename of written HDF5 file
%   jsonFile  : filename of written JSON file
%   spyInfo   : spy.SyncopyInfo object with metainformation in JSON file
%
% See also ft_save_spy

version = '0.1a';

p = inputParser;
p.addRequired('filename', ...
    @(x)validateattributes(x,{'char', 'string'},{'nonempty', 'scalartext'},'','FILENAME'));
p.addRequired('data', ...
    @(x)validateattributes(x,{'numeric'},{'nonempty', '2d'},'','DATA'));
p.addRequired('trialdefinition', ...
    @(x)validateattributes(x,{'numeric'},{'nonempty', '2d'},'','TRIALDEFINITION'));
p.addRequired('log', ...
    @(x)validateattributes(x,{'char', 'string'},{'nonempty', 'scalartext'},'','LOG'));
p.addRequired('samplerate', ...
    @(x)validateattributes(x,{'numeric'},{'nonempty', 'scalar'},'','SAMPLERATE'));
p.addRequired('channel', ...
    @(x)validateattributes(x,{'cell'},{'nonempty', '2d'},'','CHANNEL'));
p.addRequired('dimord', ...
    @(x)validateattributes(x,{'cell'},{'nonempty', '2d'},'','DIMORD'));
p.addOptional('dclass', 'AnalogData');
p.addOptional('cfg', []);

p.parse(filename, data, trialdefinition, log, samplerate, channel, dimord, varargin{:});

dclass = p.Results.dclass;
cfg = p.Results.cfg;

switch dclass
    case 'AnalogData'
        ext = '.analog';
    otherwise
        error('Currently not supported Syncopy data class %s', dclass)
end

[path, base, ~] = fileparts(filename);

%% HDF5 data file
hdfFile = fullfile(path, [base, ext]);

% datasets
dataSize = size(data);
dataSize = dataSize(end:-1:1);

delete(hdfFile)
h5create(hdfFile, ['/' dclass], dataSize, 'Datatype', class(data) )
h5write(hdfFile, ['/' dclass], permute(data, ndims(data):-1:1))

trlSize = size(trialdefinition);
trlSize = trlSize(end:-1:1);
h5create(hdfFile, '/trialdefinition', trlSize, 'Datatype', class(trialdefinition))
h5write(hdfFile, '/trialdefinition', (trialdefinition-1)')

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

jsonFile = fullfile(path, [base, ext, '.info']);
spyInfo = spy.SyncopyInfo();

spyInfo.filename = hdfFile;
spyInfo.log = log;
spyInfo.version = version;
spyInfo.dimord = dimord;
spyInfo.samplerate = samplerate;
spyInfo.type = dclass;
spyInfo.channel = channel;
spyInfo.file_checksum = hdfHash;
spyInfo.checksum_algorithm = 'openssl_sha1';
spyInfo.data_dtype = spy.dtype_mat2py(data);
spyInfo.data_shape = size(data);
spyInfo.data_offset = h5getoffset(hdfFile, ['/' dclass]);
spyInfo.trl_shape = size(trialdefinition);
spyInfo.trl_dtype = spy.dtype_mat2py(trialdefinition);
spyInfo.trl_offset = h5getoffset(hdfFile, '/trialdefinition');
spyInfo.cfg = [];
spyInfo.cfg.previous = cfg;
spyInfo.cfg.function = 'write_spy';
spyInfo.cfg.time = datestr(now);
spyInfo.cfg.user = char(java.lang.System.getProperty('user.name'));
spyInfo.cfg.hostname = char(java.net.InetAddress.getLocalHost().getHostName());
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

function offsetBytes = h5getoffset(filename, dataset)

fid = H5F.open(filename);
dset_id = H5D.open(fid, dataset);
offsetBytes = H5D.get_offset(dset_id);
H5D.close(dset_id);
H5F.close(fid);


