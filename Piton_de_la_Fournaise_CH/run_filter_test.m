clear all
close all

addpath './LibSismo'


NyF = 4.0/2.0;
FFI = 0.2;
FFE = 1.5;
[Fb, Fa]=butter(2,[FFI/NyF FFE/NyF],'bandpass');

save test_data/filter_coef.mat Fb Fa NyF FFI FFE

wfm = readsac('test_data/IU*MXZ*sac');

fdata_01 = filter(Fb, Fa, wfm.trace);
fdata_02 = flipud(filter(Fb, Fa, flipud(fdata_01)));
%fdata_03 = filtfilt(Fb, Fa, wfm.trace);
fdata_03 = l2filter(Fb, Fa, wfm.trace);

wfm = readsac('test_data/events/event_01*HHZ*SAC');

fdata_04 = filter(Fb, Fa, wfm.trace);
fdata_05 = flipud(filter(Fb, Fa, flipud(fdata_04)));
%fdata_06 = filtfilt(Fb, Fa, wfm.trace);
fdata_06 = l2filter(Fb, Fa, wfm.trace);

save test_data/filtered_data.mat fdata_01 fdata_02 fdata_03 fdata_04 fdata_05 fdata_06

