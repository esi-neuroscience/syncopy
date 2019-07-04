
% add Fieldtrip to path if on ESI Linux cluster
if isfolder('/mnt/hpx/opt/fieldtrip_github/')
    addpath('/mnt/hpx/opt/fieldtrip_github/')
    ft_defaults
end

addpath('..')
clear


%% test low-level functions load_spy and save_spy

filename = 'matlab-testdata_test';

% 5 s, 10 channels
nSamples = 5000;
nChannels = 10;
generated.data = single(rand(nSamples, nChannels));
generated.samplerate = 1000;

trl(:,1) = 1:500:nSamples;
trl(:,2) = 1:500:nSamples;
trl(:,3) = 0;
trl = int64(trl);
generated.trl = trl;

generated.dimord = {'time', 'channel'};
generated.log = 'Created some test data';
generated.channel = cell(1, nChannels);
for iChannel = 1:nChannels
    generated.channel{iChannel} = sprintf('channel_%02d', iChannel);
end


delete([fullfile(filename) '.*'])

[datFile, jsonFile, generated.spyInfo] = spy.save_spy(filename, ...
    generated.data, generated.trl, ...
    generated.log, generated.samplerate, ...
    generated.channel, generated.dimord);

% load data and compare
loaded = [];
[loaded.data, loaded.trl, loaded.spyInfo] = spy.load_spy(fullfile([filename '.ang']));

% compare generated and loaded INFO
assert(isequal(generated.spyInfo, loaded.spyInfo))

% compare HDF attributes
assert(isequal(generated.data, loaded.data))
assert(isequal(generated.trl, loaded.trl))



%% write Fieldtrip raw data to SPY

if exist('ft_defaults', 'file') == 2
    
    % create Fieldtrip data struct
    data = [];
    data.label = {'channel1', 'channel2'};
    data.fsample = 1000;
    data.trial = {rand(2, 3000), rand(2,3200)};
    data.time = cellfun(@(x) (0:length(x)-1)/data.fsample, data.trial, 'unif', false);
    data.dimord = '{rpt}_label_time';
    data.sampleinfo = [1 3000; ...
        3001 6200];
    data = ft_checkdata(data, 'datatype', 'raw', 'hassampleinfo', 'yes');
    data.trialinfo = [65 34 1; 69 25.3 2];
    
    cfg = [];
    cfg.filename = 'ft_testdata.ang';
    
    % save data to file
    spy.ft_save_spy(cfg, data);
    
    % read data back in
    loadedData = spy.ft_load_spy(cfg.filename);
    
    % test and compare only fields that exist in generated data as Fieldtrip
    % may add additional fields on load.
    
    fields = fieldnames(data);
    for iField = 1:length(fields)
        name = fields{iField};
        assert(isequal(data.(name), loadedData.(name)), ...
            'Mismatch in field %s between generated and loaded data', ...
            name)
    end
    
end



