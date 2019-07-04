function dtype = dtype_mat2py(matType)

if ~ischar(matType)
    matType = class(matType);
end


switch matType
    case 'single'
        dtype = 'float32';
    case 'double'
        dtype = 'float64';
    case 'logical'
        dtype = 'bool';
    case {'uint8', 'int8', ...
            'uint16', 'int16', ...
            'uint32', 'int32', ...
            'uint64', 'int64'}
        dtype = matType;
    otherwise
        error('Unknown data type %s', matType)
end


