function displaySAC=ds(F,varargin)
%
% ds -- Display SAC variable with specified fields only
%
% This program outputs only those fields of a SAC-file-variable
%    that are either default "c0" or precised in "varargin".
%    -any field will be output only once;
%    -if you include "-" before a field name, it will be omitted.
% "varargin" can be a cell: in this case, please precise if you want
%    all elements ( varargin{:} ), otherwise it will not work
%
% György Hetényi
% 11 Jan 2005
% Revised 16 Feb 2005
%-------------------------------------------------------------------


% PART 1: setup and input-------------------------------------------
% Basic statements
clear c0 c c2 used unwanted final lc0 lc1 c0 c1 j k l 
SAC=inputname(1);
cgood={'delta','depmin','depmax','scale','odelta','b','e','o','a','t0','t1','t2','t3','t4','t5','t6','t7','t8','t9','f','resp0','resp1','resp2','resp3','resp4','resp5','resp6','resp7','resp8','resp9','stla','stlo','stel','stdp','evla','evlo','evel','evdp','mag','user0','user1','user2','user3','user4','user5','user6','user7','user8','user9','dist','az','baz','gcarc','depmen','cmpaz','cmpinc','xminimum','xmaximum','yminimum','ymaximum','nzyear','nzjday','nzhour','nzmin','nzsec','nzmsec','sec','nvhdr','norid','nevid','npts','nwfid','nxsize','nysize','iftype','idep','iztype','iinst','istreg','ievreg','ievtyp','iqual','isynth','imagtyp','imagsrc','leven','lpspol','lovrok','lcalda','kstnm','kevnm','khole','ko','ka','kt0','kt1','kt2','kt3','kt4','kt5','kt6','kt7','kt8','kt9','kf','kuser0','kuser1','kuser2','kcmpnm','knetwk','kdatrd','kinst'};

% Default fields to output (+'trace' manually)
%c0={'delta' 'b' 'o' 't0' 't1' 't2' 't3' 't4' 'stla' 'stlo' 'stel' 'evla' 'evlo' 'evdp' 'user0' 'user1' 'user2' 'dist' 'az' 'baz' 'gcarc' 'cmpaz' 'cmpinc' 'nzyear' 'nzjday' 'nzhour' 'nzmin' 'sec' 'npts' 'kstnm' 'kevnm' 'kt0' 'kt1' 'kt2' 'kt3' 'kt4' 'kuser0' 'kuser1' 'kuser2' 'kcmpnm' 'kinst'};
c0={'delta' 'b'};
lc0=length(c0);

% Optional field to output
c1_0={varargin};
lc1_0=length(c1_0{:});
if lc1_0>0
   for i=1:lc1_0
      c1{i}=c1_0{1}{i};
   end
   lc1=length(c1);
else
   lc1=0;
end


% PART 2: verification of fields with 'cgood' cells-----------------
% Second check if a field have not been precised more than once
% Third check for minus sign and fields to omit
k=0; j=0; l=0;
used(1)=0; unwanted(1)=0;

% Original cell
for i=1:lc0
   iok=strmatch(c0{i},cgood,'exact');
   if length(iok)>0
      check=find(used==iok);
      if length(check)==0
         k=k+1;
         used(k)=iok;
      end
   end
%   if char(c1{i}(1))=='-'
%      word=char(c1{i}(2:length(c1{i})));
%      iok2=strmatch(word,cgood,'exact');
%      if length(iok2)>0
%         j=j+1;
%         unwanted(j)=iok2;
%      end
%   end
end

% Input cell
for i=1:lc1
   iok=strmatch(c1{i},cgood,'exact');
   if length(iok)>0
      check=find(used==iok);
      if length(check)==0
         k=k+1;
         used(k)=iok;
      end
   end
   if char(c1{i}(1))=='-'
      word=char(c1{i}(2:length(c1{i})));
      iok2=strmatch(word,cgood,'exact');
      if length(iok2)>0
         j=j+1;
         unwanted(j)=iok2;
      end
   end
end

% Merge used (kept indices) with unwanted indices
final=[];
for i=1:length(used)
   ifinal=find(used(i)==unwanted);
   if length(ifinal)==0
      l=l+1;
      final(l)=used(i);
   end
end

% Select final's indices from cgood
if length(final)<=0 disp('"ds":Please spcify at least one field to output!'); return; end
for i=1:length(final)
   c{i}=cgood(final(i));
end


% PART 3: output----------------------------------------------------
% Create equal length text-array with field names
lch=10;
for i=1:length(c)
   if length(char(c{i}))<lch
      dl=lch-length(char(c{i}));
      c2{i}=[char(zeros(1,dl)+32) char(c{i})];
   end
end
dl=lch-length('trace');
out_trace=[char(zeros(1,dl)+32) 'trace'];


% Output results
disp([SAC '='])
for i=1:length(c)
   disp([c2{i},': ',num2str(F.(char(c{i})))])
end
tempF=F.trace;
whosF=whos('tempF');
disp([out_trace,': [',num2str(whosF.size(1)),'x',num2str(whosF.size(2)),' ',whosF.class,']'])
%-------------------------------------------------------------------
