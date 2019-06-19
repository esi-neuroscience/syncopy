classdef SyncopyInfo
    % Class for required fields in Syncopy INFO file
    
    properties
        dimord
        version
        log
        cfg
        data
        data_dtype
        data_shape
        data_offset
        trl_dtype
        trl_shape
        trl_offset
        type
        channel
        samplerate
        hdr
    end
    
    methods
        function obj = set.dimord(obj, value)
            assert(iscell(value), 'dimord must be cell array of strings')
            assert(all(cellfun(@ischar, value)), 'dimord must be cell array of strings')
            obj.dimord = value;
        end
        
        function obj = set.version(obj, value)
            assert(ischar(value), 'version must be a string')
            obj.version = value;
        end
        
        % FIXME: implement other set functions as sanity checks
        
        function obj = set.data_dtype(obj, value)
            switch value
                case {'double', 'float64'}
                    value = 'float64';
                case {'single', 'float32'}
                    value = 'float32';
                case {'uint8', 'uint16', 'uint32', 'uint64', ...
                        'int8', 'int16', 'int32', 'int64'}
                    
                otherwise
                    error('Unsupported data type %s', value)
            end
            obj.data_dtype = value;
        end
        
        function obj = set.trl_dtype(obj, value)
            switch value
                case 'double'
                    value = 'float64';
                case 'single'
                    value = 'float32';
                case {'uint8', 'uint16', 'uint32', 'uint64', ...
                        'int8', 'int16', 'int32', 'int64'}
                    
                otherwise
                    error('Unsupported data type %s', value)
            end
            obj.trl_dtype = value;
        end
        
        function write_to_file(obj, filename)
            try % try using jsonlab
                savejson('', struct(obj), filename);
            catch me
                if strcmp(me.identifier, 'MATLAB:UndefinedFunction')
                jsonStr = jsonencode(obj);
                jsonStr = strrep(jsonStr, ',', sprintf(',\r'));
                jsonStr = strrep(jsonStr, '[{', sprintf('[\r{\r'));
                jsonStr = strrep(jsonStr, '}]', sprintf('\r}\r]'));
                fid = fopen(filename, 'w');
                fprintf(fid, jsonStr);
                fclose(fid);
                end
            end
            
        end
        
    end
end