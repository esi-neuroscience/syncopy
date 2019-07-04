function ft_save_spy(cfg, datain)
% ft_write_spy Write Fieldtrip raw data to Syncopy SPY files
%
% 
%
% See also ft_datatype_raw, spy.save_spy

% these are used by the ft_preamble/ft_postamble function and scripts
% ft_revision = '$Id$';
ft_nargin   = nargin;
ft_nargout  = nargout;

ft_defaults                   % this ensures that the path is correct and that the ft_defaults global variable is available
ft_preamble init              % this will reset ft_warning and show the function help if nargin==0 and return an error
ft_preamble debug             % this allows for displaying or saving the function name and input arguments upon an error
ft_preamble loadvar    datain % this reads the input data in case the user specified the cfg.inputfile option
ft_preamble provenance datain % this records the time and memory usage at the beginning of the function
ft_preamble trackconfig       % this converts the cfg structure in a config object, which tracks the cfg options that are being used

if ft_abort
  return
end

datain = ft_checkdata(datain, 'datatype', 'raw', ...
    'feedback', 'yes', 'hassampleinfo', 'yes', ...
    'dimord', '{rpt}_label_time');

cfg = ft_checkconfig(cfg, 'required', {'filename'});

filename = ft_getopt(cfg, 'filename');        % there is no default

[folder, basename, ext] = fileparts(filename);
if ~strcmp(ext, '.ang')
    filename = [filename '.ang'];
end

dimord = tokenize(datain.dimord, '_');
dimord(strcmp(dimord, '{rpt}')) = '';
dimord{strcmp(dimord, 'label')} = 'channel';


nTrialinfocols = 0;
if isfield(datain, 'trialinfo')
    nTrialinfocols = size(datain.trialinfo, 2);
end

trl = zeros(length(datain.trial), 3+nTrialinfocols);

indx = 1;
for iTrial = 1:length(datain.trial)
    trl(iTrial, 1) = indx;
    trl(iTrial, 2) = indx+length(datain.time{iTrial})-1;
    trl(iTrial, 3) = round(datain.time{iTrial}(1) * datain.fsample);  
    indx = indx + length(datain.time{iTrial});
end
if isfield(datain, 'trialinfo')
    trl(:, 4:6) = datain.trialinfo;
end

log = 'Created some test data';
version = '0.1a';


spy.save_spy(filename, ...
    cat(2, datain.trial{:}), trl, ...
    log, datain.fsample, ...
    datain.label, dimord);

ft_postamble debug               % this clears the onCleanup function used for debugging in case of an error
ft_postamble trackconfig         % this converts the config object back into a struct and can report on the unused fields
ft_postamble previous   datain   % this copies the datain.cfg structure into the cfg.previous field. You can also use it for multiple inputs, or for "varargin"
ft_postamble provenance % this records the time and memory at the end of the function, prints them on screen and adds this information together with the function name and MATLAB version etc. to the output cfg
ft_postamble history    % this adds the local cfg structure to the output data structure, i.e. dataout.cfg = cfg


end