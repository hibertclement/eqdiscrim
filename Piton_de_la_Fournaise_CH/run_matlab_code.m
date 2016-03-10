clear all
close all

addpath './LibSismo'

zwfm = readsac('test_data/events/event_01*HHZ*SAC');
nwfm = readsac('test_data/events/event_01*HHN*SAC');
ewfm = readsac('test_data/events/event_01*HHE*SAC');

values = cell(3, 1);
values{1} = zwfm.trace;
values{2} = nwfm.trace;
values{3} = ewfm.trace;

[att, att_std] = ComputeAttributes(values, 100, 3);

save test_data/event01.mat att
