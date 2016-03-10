function dat=datenumref(A)
% returns the reference time of a SAC file in the classic format of datenum
% Revised on 02/05/2005 (J. Vergne)
%   - loop over multiple files

for i=1:length(A)
    [mois,jour]=jd2md(A(i).nzjday,A(i).nzyear);
    dat(i)=datenum(A(i).nzyear,mois,jour,A(i).nzhour,A(i).nzmin,A(i).sec);
end
