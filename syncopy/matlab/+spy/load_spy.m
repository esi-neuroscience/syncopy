%
% Load data from SynCoPy containers
% 
% Created: 2019-04-24 16:40:56
% Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
% Last modification time: <2019-04-30 16:16:43>

function [data, trl, attrs, json] = load_spy(in_name, varargin)
    % LOAD_SPY Load data from HDF5/JSON SPY container 
    
    % Import custom exceptions
    import spy.utils.spy_error;
    
    % Default values for optional inputs
    fname = '';    

    % Parse inputs 
    valid_fsloc = @(x) validateattributes(x, {'char'}, {'scalartext', 'nonempty'});
    p = inputParser;
    addRequired(p, 'in_name', valid_fsloc);
    addParameter(p, 'fname', fname, valid_fsloc);
    parse(p, in_name, varargin{:});
    
    % Make sure `in_name` is a valid filesystem-location: in case 'dir' and
    % 'dir.spy' exists, preference will be given to 'dir.spy'
    [~, ~, ext] = fileparts(p.Results.in_name);
    if isempty(ext) == 0
        in_spy = [in_name, '.spy'];
        if isempty(what(in_spy))
            in_spy = in_name;
        end
    else
        in_spy = in_name;
    end
    w_spy = what(in_spy);
    if isempty(w_spy)
        spy_error(['Cannot read ', in_name, ': object does not exist. '], 'io');
    end
    in_name = w_spy.path;       % get absolute path of provided spy-dir
    
    % Either (try to) load newest fileset or look for a specific one
    if isempty(p.Results.fname)
        
        % Get most recent json file in `in_name` (abort if we don't find one)
        files = dir(fullfile(in_name, '*.info'));
        if isempty(files)
            spy_error(['Cannot find .info file in ', in_name], 'io')
        end
        [~, idx] = max([files.datenum]);
        in_file = fullfile(in_name, files(idx).name);
        
    else

        % Remove (if any) path as well as extension from provided file-name(-pattern)
        % and convert `fname` to search pattern if it does not already conatin wildcards
        [~, in_base, in_ext] = fileparts(p.Results.fname);
        fname = [in_base, in_ext];
        if length(strfind(fname, '*')) == 0
            fname = ['*', fname, '*'];
        end
        in_ext = strrep(in_ext, '*', '');
        if isempty(in_ext)
            expected_count = 2;
        else
            validatestring(in_ext, {'.info', '.dat'}, mfilename, 'fname')
            expected_count = 1;
        end

        % Abort in case we don't find our expected number of files (either one or two)
        files = dir(fullfile(in_name, fname));
        in_count = length(files);
        if in_count ~= expected_count
            msg = ['Expected ', num2str(expected_count), ' files in ', in_name];
            msg = [msg, ' found ', num2str(in_count)];
            spy_error(msg, 'value')
        end
        in_file = fullfile(in_name, files(1).name);
        
    end

    % Build struct of files to read from 
    [in_path, in_base, ~] = fileparts(in_file);
    in_files.info = fullfile(in_path, [in_base, '.info']);
    in_files.dat = fullfile(in_path, [in_base, '.dat']);
    
    
    % Get content info of dat-HDF5 container 
    h5toc = h5info(in_files.dat);
    dset_names = {h5toc.Datasets.Name};
    msk = ~strcmp(dset_names, 'trialdefinition');
    dclass = dset_names{msk};
    ndim = length(h5toc.Datasets(msk).Dataspace.Size);
    
    % Account for C-ordering by permuting contents of dataset
    data = permute(h5read(in_files.dat, ['/', dclass]), [ndim : -1 : 1]);

    % The `trialdefinition` part is always 2D -just transpose it
    trl = h5read(in_files.dat, '/trialdefinition')';
    
    % extract container attributes
    h5attrs = h5toc.Attributes;
    
    % read json file and compare to HDF5 attributes
    json = jsondecode(fileread(in_files.info));
    
    attrs = struct('name', cell(size(h5attrs)), ...
        'value', cell(size(h5attrs)));
    for iAttr = 1:length(h5attrs)
        name = h5attrs(iAttr).Name;
        value = h5attrs(iAttr).Value;
        if iscell(value) && numel(value) == 1
            value = value{1};
        end
        attrs(iAttr).(name) = value;        
        assert(isequal(json.(name), value), ...
            'JSON/HDF5 mismatch for attribute %s', name)
    end
    
    % FIXME: check other json attributes: data_dtype, data_shape, ...
        
    return
end
