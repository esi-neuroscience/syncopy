% -*- coding: utf-8 -*-
%
% Load data from SynCoPy containers
% 
% Created: 2019-04-24 16:40:56
% Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
% Last modification time: <2019-04-25 17:27:53>

function load_spy(in_name, varargin)
    
    import spy.utils.spy_error;
    
    % Default values for optional inputs
    fname = '';
    checksum = false;

    % Parse inputs 
    valid_fsloc = @(x) validateattributes(x, {'char'}, {'scalartext', 'nonempty'});
    p = inputParser;
    addRequired(p, 'in_name', valid_fsloc);
    addParameter(p, 'fname', fname, valid_fsloc);
    addParameter(p, 'checksum', checksum, ... 
                 @(x) validateattributes(x, {'numeric'}, {'scalar', 'binary'}));
    parse(p, in_name, varargin{:});
    
    % Make sure `in_name` is a valid filesystem-location: in case 'dir' and
    % 'dir.spy' exists, preference will be given to 'dir.spy'
    [~, ~, ext] = fileparts(p.Results.in_name);
    if length(ext) == 0
        in_spy = [in_name, '.spy'];
        if length(what(in_spy)) == 0;
            in_spy = in_name;
        end;
    else
        in_spy = in_name;
    end
    w_spy = what(in_spy);
    if length(w_spy) == 0
        spy_error(['Cannot read ', in_name, ': object does not exist. '], 'io');
    end
    in_name = w_spy.path;       % get absolute path of provided spy-dir
    
    % Either (try to) load newest fileset or look for a specific one
    if length(p.Results.fname) == 0
        
        % Get most recent json file in `in_name` (abort if we don't find one)
        files = dir(fullfile(in_name, '*.info'));
        if length(files) == 0
            spy_error(['Cannot find .info file in ', in_name], 'io')
        end
        [~, idx] = max([files.datenum]);
        in_file = fullfile(in_name, files(idx).name)
        
    else

        % Remove (if any) path as well as extension from provided file-name(-pattern)
        % and convert `fname` to search pattern if it does not already conatin wildcards
        [~, in_base, in_ext] = fileparts(p.Results.fname);
        fname = [in_base, in_ext];
        if length(strfind(fname, '*')) == 0
            fname = ['*', fname, '*'];
        end
        in_ext = strrep(in_ext, '*', '');
        if length(in_ext) == 0
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
    
    % jsondecode(fileread(in_files.info))
    
    keyboard;
