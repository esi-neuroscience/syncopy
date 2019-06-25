%
% Load data from SynCoPy containers
% 
% Created: 2019-04-24 16:40:56
% Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
% Last modification time: <2019-04-30 16:16:43>

function [data, trl, attrs, json] = load_spy(in_file)
    % LOAD_SPY Load data from HDF5/JSON SPY container 
    
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
