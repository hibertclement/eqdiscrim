clear all
close all

addpath './LibSismo'

zwfm = readsac('test_data/pol*MXZ*sac');
rwfm = readsac('test_data/pol*MXR*sac');
twfm = readsac('test_data/pol*MXT*sac');

SIG = cell(3, 1);
SIG{1} = zwfm.trace;
SIG{2} = twfm.trace;
SIG{3} = rwfm.trace;

HeavySmoothCoef=4;
ENV=abs(hilbert(SIG{1}));
SMOOTHEDENVELOP=filter(ones(HeavySmoothCoef,1)./HeavySmoothCoef,1,ENV);
[MT2,ImaxT2]=max(SMOOTHEDENVELOP);
EndWindow=round(ImaxT2/3);

xP=SIG{3}(1:EndWindow)';% Horiz 1
yP=SIG{2}(1:EndWindow)';% Horiz 2
zP=SIG{1}(1:EndWindow)';% Vertical (Up)
MP=cov([xP; yP; zP]');%covariance(xP,yP,zP);
[pP,DP] = eig(MP);
%%% DP contains the eigenvalues of the covariance matrix, with
%%% DP(3,3)>DP(2,2)>DP(1,1)
%%% pP contains the eigenvectors, where the first column is the
%%% eigenvector that corresponds to the smallest eigenvalue, the
%%% second one to the intermedian eigenvalue and the third one to
%%% the largest eigenvalue (this one shows the dominant particle motion)
rectilinP=1-((DP(1,1)+DP(2,2))/(2*DP(3,3)));% Rectinilarity
azimuthP=atan(pP(2,3)/pP(1,3))*180/pi;  % Azimuth - Not Necessary?
dipP=atan(pP(3,3)/sqrt(pP(2,3)^2+pP(1,3)^2))*180/pi; % Dip
Plani=1-(2*DP(1,1))/((DP(3,3)+DP(2,2))); % Planarity
  
save test_data/pol_test.mat