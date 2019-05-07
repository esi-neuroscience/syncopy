% 
% ALREADY KNOW YOU THAT WHICH YOU NEED
% 
% Created: 2019-04-25 15:07:03
% Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
% Last modification time: <2019-04-30 13:58:06>

% Add SynCoPy package to MATLAB path
spy_path = what(['..', filesep, 'matlab']);
addpath(spy_path.path);

% spy.load_spy('adata');
[data, trl, attrs] = spy.load_spy('adata', 'fname', 'adata');
% spy.load_spy('test.spy', 'fname', 'ttwet');
