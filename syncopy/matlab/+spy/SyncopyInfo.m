classdef SyncopyInfo
    % Class for required fields in Syncopy INFO file
       
    properties
        filename
        dimord
        version
        log
        cfg
        data_dtype
        data_shape
        data_offset
        data_checksum
        checksum_algorithm
        trl_dtype
        trl_shape
        trl_offset
        type
        channel
        samplerate
        hdr
    end
    
    properties (Access = private, Constant = true, Hidden=true)
        supportedDataTypes = {'int8', 'uint8', 'int16', 'uint16', ...
            'int32', 'uint32', 'int64', 'uint64', ...
            'float32', 'float64', ...
            'complex64', 'complex128'};
        requiredFields = {'dimord', 'version', 'log', 'cfg', ...
            'data_dtype', 'data_shape', 'data_offset', ...
            'trl_dtype', 'trl_shape', 'trl_offset'}
    end
    
    methods
        
        function obj = SyncopyInfo(infoStruct)
            
            if nargin > 0
                
                if ischar(infoStruct)
                    assert(exist(infoStruct, 'file') == 2)
                    infoStruct = spy.jsonlab.loadjson(infoStruct);
                end
                
                fields = fieldnames(infoStruct);
                
                for iField = 1:length(fields)
                    name = fields{iField};
                    
                    if ~isprop(obj, name) || isempty(infoStruct.(name))
                        continue
                    end
                    
                    obj.(name) = infoStruct.(name);
                    
                end
                
            end
            
        end
        
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
            assert(ismember(value, obj.supportedDataTypes), ...
                'Unsupported dtype %s', value)
            obj.data_dtype = value;
        end
        
        function obj = set.trl_dtype(obj, value)
            assert(ismember(value, obj.supportedDataTypes), ...
                'Unsupported dtype %s', value)
            obj.trl_dtype = value;
        end
        
        function output_struct = struct(obj)
            output_struct = obj.obj2struct(obj);            
            
        end
        
        function hasAllRequired = assert_has_all_required(obj)
            hasAllRequired = true;
            
            for iRequired = 1:length(obj.requiredFields)
                name = obj.requiredFields{iRequired};
                assert(~isempty(obj.(name)), ...
                    'Required field %s is not set', name)
            end
            
        end
        
        function write_to_file(obj, filename)
            obj.assert_has_all_required();
            spy.jsonlab.savejson('', struct(obj), filename);
        end
        
    end
    
    methods ( Static = true )
        function output_struct = obj2struct(obj)
            properties = fieldnames(obj); % works on structs & classes (public properties)
            
            for i = 1:length(properties)
                val = obj.(properties{i});
                
                if ~isstruct(val) &&~isobject(val)
                    output_struct.(properties{i}) = val;
                else
                    
                    if isa(val, 'serial') || isa(val, 'visa') || isa(val, 'tcpip')
                        % don't convert communication objects
                        continue
                    end
                    
                    temp = obj.obj2struct(val);
                    
                    if ~isempty(temp)
                        output_struct.(properties{i}) = temp;
                    end
                    
                end
                
            end
            
        end
    end
    
end
