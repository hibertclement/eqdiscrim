function O=bpf(I,fmin,fmax,ordre,tau)
% function O=bpf(I,fmin,fmax,ordre,tau)
%
% Butterworth-type low-pass filter (filtfilt) of a signal "I" (SAC format
%   possible), between cutting frequencies "fmin" and "fmax" at
%   order "ordre" (default=4)
% If "I" is in SAC format, "O" will also be in SAC format and "tau" is
%   not necessary. Example: O=bpf(I,1)
% If "I" is a vector, "O" is also a vector and "tau" is needed.
%   Example: O=bpf(I,1,3)
%
% 11 Jan 2005

is_filtfilt=1;

if nargin<3
	disp('"bpf": Missing parameters')
	quit
end	
if nargin<4 & isstruct(I)
	ordre=4;
end	
if nargin<4 & ~isstruct(I)
	disp('"bpf": Missing parameters')
	quit
end

if isstruct(I)
	O=I;
	fe=1/(2*I.delta);
	[b,a]=butter(ordre,[fmin/fe fmax/fe]);
	if is_filtfilt==0
		O.trace=filter(b,a,I.trace);
	else
                O.trace=filtfilt(b,a,I.trace);
	end
else
	fe=1/(2*tau);
	[b,a]=butter(ordre,[fmin/fe fmax/fe]);
	if is_filtfilt==0 
		O=filter(b,a,I);
	else
                O=filtfilt(b,a,I);
	end
end
