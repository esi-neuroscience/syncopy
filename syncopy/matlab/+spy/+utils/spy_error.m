% 
% Custom package-specific errors for SyNCoPy
% 
% Created: 2019-04-25 14:08:55
% Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
% Last modification time: <2019-04-25 15:28:03>

function spy_error(msg, identifier)
    
    % Parse inputs 
    p = inputParser;
    addRequired(p, 'msg', @(x) validateattributes(x, {'char'}, {'scalartext', 'nonempty'}));
    addRequired(p, 'identifier', @(x) any(validatestring(x, {'value', 'type', 'io'})));
    parse(p, msg, identifier);

    % Build MATLAB exception and throw it as if it occurred within caller
    switch p.Results.identifier
      case 'type'
        msgid = 'SyNCoPy:TypeError';
      case 'value'
        msgid = 'SyNCoPy:ValueError';
      case 'io'
        msgid = 'SyNCoPy:IOError';
    end
    throwAsCaller(MException(msgid, p.Results.msg));
