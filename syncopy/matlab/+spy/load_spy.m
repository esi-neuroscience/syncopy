function [data, trl, attrs, json] = load_spy(in_file)
    % LOAD_SPY Load data from HDF5/JSON SPY container 
    %
    %   [data, trl, h5attrs, json] = load_spy(in_file)
    %

    infoFile = fullfile([in_file '.info']);
       
    % Get content info of dat-HDF5 container 
    h5toc = h5info(in_file);
    dset_names = {h5toc.Datasets.Name};
    msk = ~strcmp(dset_names, 'trialdefinition');
    dclass = dset_names{msk};
    ndim = length(h5toc.Datasets(msk).Dataspace.Size);
    
    % Account for C-ordering by permuting contents of dataset
    data = permute(h5read(in_file, ['/', dclass]), [ndim : -1 : 1]);

    % The `trialdefinition` part is always 2D -just transpose it
    trl = h5read(in_file, '/trialdefinition')' + 1;
    
    % extract container attributes
    h5attrs = h5toc.Attributes;
    
    % read json file and compare to HDF5 attributes
    json = spy.jsonlab.loadjson(infoFile);
    
    for iAttr = 1:length(h5attrs)
        name = h5attrs(iAttr).Name;
        value = h5attrs(iAttr).Value;
        if iscell(value) && numel(value) == 1
            value = value{1};
        end
        if ~ischar(value)
            value = value';
        end
        attrs.(name) = value;        
        assert(isequal(json.(name), value), ...
            'JSON/HDF5 mismatch for attribute %s', name)
    end
    
    % FIXME: check other json attributes: data_dtype, data_shape, ...
        
    return
end
