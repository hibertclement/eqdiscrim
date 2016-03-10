function jd=md2jd(mo,day,yr)
%  function jd=md2jd(mo,day,<yr>)
%
%  Function to transfer date to Julian day with the option to have leap years
%  Input 'leap' or year number for <yr> if you want to use leap years
%  Default is regular year
%
%  Remark: a leap year is a year with 366 days: every 4th year is a leap year
%  BUT every 100th is NOT _AND_ every 400th _IS_ again a leap year...
%
% György Hetényi
% 22 Nov 2004
% ------------------------------------------------------------------

for i=1:length(mo)

if nargin<3 yr(i)=1; end
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
   
if day(i)>mos(mo(i)) disp('Error: day is > than possible!')
   return
else
   jd(i)=sum(mos(1:mo(i)-1))+day(i);
end


end
