
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

[datFile, jsonFile, generated.spyInfo] = spy.write_spy(filename, ...
    generated.data, generated.trl, ...
    generated.log, generated.samplerate, ...
    generated.version, generated.channel, generated.dimord);

% load data and compare
loaded = [];
[loaded.data, loaded.trl, loaded.attrs, loaded.json] = spy.load_spy(fullfile([filename '.ang']));

% compare generated and loaded INFO
loaded.spyInfo = spy.SyncopyInfo(jsonFile);
assert(isequal(generated.spyInfo, loaded.spyInfo))

% compare HDF attributes
assert(isequal(generated.data, loaded.data))
assert(isequal(generated.trl, loaded.trl))
assert(isequal(generated.log, loaded.attrs.log))
assert(isequal('AnalogData', loaded.attrs.type))
assert(isequal(generated.dimord, loaded.attrs.dimord))
assert(isequal(generated.version, loaded.attrs.version))
assert(isequal(generated.channel, loaded.attrs.channel))


%% write Fieldtrip raw data to SPY

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

spy.ft_write_spy(cfg, data)

[data, trl, attrs, json] = spy.load_spy([cfg.filename, '.ang']);


% spy.ft_write_spy

