function hp=plots(T,s,t0,dy,ampmul,orientation)
%
% function hp=plots(T,<s>,<t0>,<dy>,<ampmul>,<orientation>)
%
% Plot of a SAC-file's trace
% with time (seconds) on the abscissa and indication
% of the absolute time of the "0" (on abscissa) corresponding
% to the reference time in SAC header
%
% T must be a SAC-structure
%
% Optionnal arguments :
% s  -> type of plot (cf plot.m) : example : 'k' or 'o' or 'ko'
% t0 -> shift (in seconds) of the first time sample
% dy -> vertical shift of the trace
% ampmul -> amplification factor (default is 1)
% orientation -> 'hori' (horizontal trace - default) ou 'vert' (vertical trace)
%
% 12 Jan 2005
% Revised 16 Feb 2005
% Remarks : use of home made function : datenumfirst and jd2md (enclosed)

tt=[0:length(T.trace)-1]*T.delta;

if nargin<2
	s='k-';
end	
if nargin<3
	t0=0;
end
if nargin<4
	dy=0;
end
if nargin<5
	ampmul=1;
end
if nargin<6
	orientation='hori';
end

timeref=datestr(datenumfirst(T)+(T.b-T.o+t0)/86400);

switch orientation
	case 'hori'
		hp=plot(tt+t0,T.trace*ampmul+dy,s);
	case 'vert'
		hp=plot(T.trace*ampmul+dy,tt+t0,s);
	otherwise
		disp('"plots": orientation is either ''hori'' or ''vert''')
end			
xlabel(['time (s) relative to ' timeref])

% ----------------------------------------------------------------------------

function dat=datenumfirst(A)
% retourne le temps du premier echantillon au format classique de datenum
% returns the first time sample in the classic format of datenum
% Revised on 02/05/2005 (J. Vergne)
%   - loop over multiple files

for i=1:length(A)
    [mois,jour]=jd2md(A(i).nzjday,A(i).nzyear);
    dat(i)=datenum(A(i).nzyear,mois,jour,A(i).nzhour,A(i).nzmin,A(i).sec);
end

% ----------------------------------------------------------------------------

function [mo,day]=jd2md(jd,yr)
%  function [mo,day]=jd2md(jd,<yr>)
%
%  Function to transfer Julian day to date with the option to have leap years
%  Input 'leap' or year number for <yr> if you want to use leap years
%  Default is regular year
%
%  Remark: a leap year is a year with 366 days: every 4th year is a leap year
%  BUT every 100th is NOT _AND_ every 400th _IS_ again a leap year...
%
%  György Hetényi
%  22 Nov 2004
% ------------------------------------------------------------------
for i=1:length(jd)

if nargin<2 yr(i)=1; end
if ischar(yr(i))==1 & yr(i)=='leap' type='leap';
elseif mod(yr(i),4)==0 & mod(yr(i),100)~=0 type='leap';
elseif mod(yr(i),400)==0 type='leap';
else type='regu';
end

clear mos
if type=='leap'
mos=[31,29,31,30,31,30,31,31,30,31,30,31];
end
if type=='regu'
mos=[31,28,31,30,31,30,31,31,30,31,30,31];
end

k=1;
while jd(i)>sum(mos(1:k))
   k=k+1;
end
mo(i)=k;
day(i)=jd(i)-sum(mos(1:mo(i)-1));
   
end

