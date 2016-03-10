function time=traveltime(phase,prof,dist,method)
% time=traveltime(phase,depth,dist,[method])
%
% This function gives the travel-time in the IASP91 model
%    for the phase "phase" (P,S,ScS,PKPab,...) and for an event
%    (or a collection of events) at given depths ("depth")
%    and epicentral distance ("dist").
% Both depth and dist could be vectors (if not the same size), one of them 
%    them must be a single value.
%
% Remark :
% ----------
% "method" is an optionnal field to indicate the interpolation method.
% By default it is 'linear' (cf. documentation of griddata.m)
% but due to unknown problem with gridata, you might change it to 'v4'
% if errors occur.
%

if nargin<4
	method='linear';
end

pp=which('traveltime');
pathIASP=[pp(1:end-12) 'IASP/'];

lof=dir([pathIASP phase '.mat']);

if ~isempty(lof)

	if length(prof)>1 & length(dist)==1
		dist=zeros(size(prof))+dist;
	end
	if length(dist)>1 & length(prof)==1
		prof=zeros(size(dist))+prof;
	end

	load([pathIASP phase '.mat']);
	TIME=IASP.time;
	[PROF,DIST]=meshgrid(IASP.depth,IASP.dist);
	time=griddata(PROF,DIST,TIME,prof,dist,method);
    idist=find(dist<IASP.dist(1) | dist>IASP.dist(length(IASP.dist)));
    time(idist)=NaN;
else
	disp(['The phase ' phase ' is not available'])
	time=zeros(size(prof))+NaN;
end
