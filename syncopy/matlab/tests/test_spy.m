
if isfolder('/mnt/hpx/opt/fieldtrip_github/')
    addpath('/mnt/hpx/opt/fieldtrip_github/')
    ft_defaults
end

addpath('..')
clear


%% generate data

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
generated.version = '0.1a';
generated.channel = cell(1, nChannels);
for iChannel = 1:nChannels
    generated.channel{iChannel} = sprintf('channel_%02d', iChannel);
end


delete([fullfile(filename) '.*'])

[datFile, jsonFile, generated.spyInfo] = spy.save_spy(filename, ...
    generated.data, generated.trl, ...
    generated.log, generated.samplerate, ...
    generated.version, generated.channel, generated.dimord);

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
    data = [];
    data.label = {'channel1', 'channel2'};
    data.fsample = 1000;
    data.trial = {rand(2, 3000), rand(2,3200)};
    data.time = cellfun(@(x) (0:length(x)-1)/data.fsample, data.trial, 'unif', false);
    data.dimord = '{rpt}_label_time';
    data.sampleinfo = [1 3000; ...
        3001 6200];
    data = ft_checkdata(data, 'datatype', 'raw', 'hassampleinfo', 'yes');
    data.trialinfo = [65 34 1; 69 25 2];
    
    cfg = [];
    cfg.filename = 'ft_testdata';
    
    spy.ft_save_spy(cfg, data);
    
    [data, trl, spyInfo] = spy.load_spy([cfg.filename, '.ang']);
end

% spy.ft_write_spy

