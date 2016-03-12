clear all
close all

addpath './LibSismo'
SpecWdow = 200;
noverlap = 0.9 * 100;

wfm = readsac('test_data/IU*MXZ*sac');

sps=4.0;
n=2^nextpow2(2*length(wfm.trace)-1);
n2=2^nextpow2(2*SpecWdow-1);

FFT = fft(wfm.trace, n);
%Freq1=linspace(0,1,n/2)*(sps/2);

[spec, F, T] = spectrogram(wfm.trace, SpecWdow, noverlap, n2, sps);
smooth_spec=filter(ones(1,100)./100,1,(abs(spec)),[],1);

save test_data/spec_test.mat spec smooth_spec F T FFT