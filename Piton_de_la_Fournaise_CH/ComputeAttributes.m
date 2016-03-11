function [ATTRIBUTES,ATTRIBUTESstd]=ComputeAttributes(SIG,sps,flag)

%; SIG : picked signal;
% sps=samplerate (Hz); 
%flag > 
% flag=1 mono-station mono channel
% flag=2 mono-station 3C
% flag=3 multi-stations
% flag=4 multi-stations 3C
%
% /!\ If 3C SIG *MUST* be SIG{1}=Up; SIG{2,3}=Horiz.

for ii=1:length(SIG)
    
%% Waveform parameters
CoefSmooth=3;% light smoothing for better estimation of the envelope feature
HeavySmoothCoef=sps;% Heavy smooth coeff
ENV{ii}=abs(hilbert(SIG{ii})); % Computation of the envelope
TesMAX(ii)=max(filter(ones(CoefSmooth,1)./CoefSmooth,1,ENV{ii})); % Max envelope
TesMEAN(ii)=mean(filter(ones(CoefSmooth,1)./CoefSmooth,1,ENV{ii}./max(ENV{ii}))); % Mean normalized envelope
TesMEDIAN(ii)=median(filter(ones(CoefSmooth,1)./CoefSmooth,1,ENV{ii}./max(ENV{ii}))); % Median normalized envelope
TesSTD(ii)=std(filter(ones(CoefSmooth,1)./CoefSmooth,1,ENV{ii}./max(ENV{ii}))); % Standard Deviation of envelope
RappMaxMean(ii)=1/(mean(TesMEAN)); % Ratio max/mean normalized envelope
RappMaxMedian(ii)=1/(mean(TesMEDIAN)); % Ratio max/mean normalized envelope
KurtoEnv(ii)=kurtosis(filter(ones(CoefSmooth,1)./CoefSmooth,1,ENV{ii}./max(ENV{ii}))); % Kurtosis of normalized smoothed envelope
KurtoSig(ii)=kurtosis(SIG{ii}./max(SIG{ii})); % Kurtosis normalized Signal
SkewnessEnv(ii)=skewness(filter(ones(CoefSmooth,1)./CoefSmooth,1,ENV{ii}./max(ENV{ii}))); % Skewness of normalized smoothed envelope
SkewnessSig(ii)=skewness(SIG{ii}./max(SIG{ii})); % Skewness normalized Signal
Duration(ii)=length(SIG{ii})./sps; % Duration of signal
SMOOTHEDENVELOP{ii}=filter(ones(HeavySmoothCoef,1)./HeavySmoothCoef,1,ENV{ii}); % heavy smoothing of the envelope for Increasgin/Decreasing estimate
[MT2,ImaxT2]=max(SMOOTHEDENVELOP{ii}); % Find max enve smoothed
TASSENCDES(ii)=((ImaxT2)/(length(SIG{ii})-ImaxT2));% ratio Increasing time/Decreasing Time

if size((1:length(SIG{ii}(ImaxT2:end))),2)==size(abs(hilbert(SIG{ii}(ImaxT2:end))),2)
    RMSDECAMPENV(ii)=sqrt(mean(abs(hilbert(SIG{ii}(ImaxT2:end)./max(SIG{ii})))-...
    (1-((1/length(SIG{ii}(ImaxT2:end))).*(1:length(SIG{ii}(ImaxT2:end)))))).^2);%RMS between decreasing of envelope and straight line
    else
    RMSDECAMPENV(ii)=sqrt(mean(abs(hilbert(SIG{ii}(ImaxT2:end)./max(SIG{ii})))-...
    (1-((1/length(SIG{ii}(ImaxT2:end))).*(1:length(SIG{ii}(ImaxT2:end)))'))).^2);%RMS between decreasing of envelope and straight line
end

[COR,lag]=xcorr(SIG{ii},'coeff'); %autocorrelation
CORSMOOTH=abs(hilbert(COR)); %autocorrelation enveloppe
CORSMOOTH=filtfilt(ones(HeavySmoothCoef,1)./HeavySmoothCoef,1,CORSMOOTH); % Smoothing
I=find(lag>=0);
INT1(ii)=trapz(CORSMOOTH(I(1)+(1:round(length(I)/3)))./max(CORSMOOTH)); % Integrale de 0 à 1/3
INT2(ii)=trapz(CORSMOOTH(round(length(I)/3)+I(1):end)./max(CORSMOOTH)); %Integrale de 1/3  à end
INT(ii)=INT1(ii)/INT2(ii); % rapport des intégrales
[CORl,CORp]=findpeaks(filtfilt(ones(100,1)./100,1,CORSMOOTH./max(CORSMOOTH)),'MinPeakHeight',0.4);
CORPEAKNUMBER(ii)=length(CORp);

% Energy in several frequency band
NyF=sps/2;% Nyquist Frequency
FilterFrequencyI=[0.1,1,3,10,20]; % to test
FilterFrequencyE=[1,3,10,20,0.99*NyF]; % to test
for jj=1:length(FilterFrequencyI) 
    [Fa,Fb]=butter(2,[FilterFrequencyI(jj)/NyF FilterFrequencyE(jj)/NyF],'bandpass'); % Butterworse Filter with normalized frequency, 1=Nyquist
    DATAF{ii,jj}=filtfilt(Fa,Fb,SIG{ii}); % acausal filterting
    ES(ii,jj)=log10(trapz(abs(hilbert(DATAF{ii,jj})))); % compute energy in several frequency band
    KurtoF(ii,jj)=kurtosis(DATAF{ii,jj});% Kurto in several Fband
end

%% Spectral features
% - Full Spectrum
NyF=sps/2;% Nyquist Frequency
n=2048;
n=2^nextpow2(2*length(SIG{ii})-1);
Freq1=linspace(0,1,n/2)*(sps/2);% Frequency array 
FFTdata{ii}=2*abs(fft(SIG{ii},n))./length(SIG{ii}).^2;% FFT modulus
SMOOTHEDFFT{ii}=(filter(ones(300,1)./300,1,FFTdata{ii}(1:n/2)));% Smoothed spectrum
NORMALIZEDSMOOTHEDFFT{ii}=SMOOTHEDFFT{ii}./(max(SMOOTHEDFFT{ii}));% Normalized spectrum
MEANFFT(ii)=mean(NORMALIZEDSMOOTHEDFFT{ii}); % Mean Normalized SPectrum = Y centroid
XCENTERFFT(ii)=sum( (1:length(NORMALIZEDSMOOTHEDFFT{ii}))' .*NORMALIZEDSMOOTHEDFFT{ii},1)...
    /sum(NORMALIZEDSMOOTHEDFFT{ii},1); % X of centroid (X mean)
XCENTERFFT1QUART(ii)=sum((1:length(NORMALIZEDSMOOTHEDFFT{ii}(1:round(XCENTERFFT(ii)))))'.*NORMALIZEDSMOOTHEDFFT{ii}(1:round(XCENTERFFT(ii))),1)...
    /sum(NORMALIZEDSMOOTHEDFFT{ii}(1:round(XCENTERFFT(ii))),1); % X 1 quartile
XCENTERFFT3QUART(ii)=sum((1:length(NORMALIZEDSMOOTHEDFFT{ii}(round(XCENTERFFT(ii)):end)))'.*NORMALIZEDSMOOTHEDFFT{ii}(round(XCENTERFFT(ii)):end),1)...
    /sum(NORMALIZEDSMOOTHEDFFT{ii}(round(XCENTERFFT(ii)):end),1)+round(XCENTERFFT(ii)); % X 3 quartile
[MAXFFT(ii),IFMAXFFT]=max(SMOOTHEDFFT{ii}); %Max of FFT
FMAX(ii)=Freq1(IFMAXFFT); % Frequence at Max(FFT)
FQUART1(ii)=Freq1(round(XCENTERFFT1QUART(ii))); %Frequence 1 quartile
FQUART3(ii)=Freq1(round(XCENTERFFT3QUART(ii))); %Frequence 3 quartile
FCENTROID(ii)=Freq1(round(XCENTERFFT(ii))); % Frequence centroid
MEDIANFFT(ii)=median(NORMALIZEDSMOOTHEDFFT{ii}); % Median Normalized FFT
VARFFT(ii)=var(NORMALIZEDSMOOTHEDFFT{ii}); % var Normalized FFT
[ValPeaks{ii},ipeaks{ii}]=findpeaks(NORMALIZEDSMOOTHEDFFT{ii},'minpeakheight',0.75);
NBRPEAKFFT(ii)=length(ipeaks{ii}); % Number of peaks in normalized fft
MEANPEAKS(ii)=mean(ValPeaks{ii}); % Mean peaks value for peaks>0.75
E1FFT(ii)=trapz(NORMALIZEDSMOOTHEDFFT{ii}(1:end/4));%Energy in the 1-12.5Hz 
E2FFT(ii)=trapz(NORMALIZEDSMOOTHEDFFT{ii}(end/4:2*end/4));%idem 12.5-25Hz
E3FFT(ii)=trapz(NORMALIZEDSMOOTHEDFFT{ii}(2*end/4:3*end/4));%idem 25-37.5Hz
E4FFT(ii)=trapz(NORMALIZEDSMOOTHEDFFT{ii}(3*end/4:end));% idem 37.5-50Hz
% Moment 
for k=0:2
    MOMENT(k+1)=sum((Freq1.^k)'.*(NORMALIZEDSMOOTHEDFFT{ii}(1:n/2).^2));
end
gamma1(ii)=MOMENT(2)/MOMENT(1); %centroid
gamma2(ii)=sqrt(MOMENT(3)/MOMENT(1)); % gyration radius
gammas(ii)=sqrt(abs(gamma1(ii)^2-gamma2(ii)^2)); % spectrum width 

% - Pseudo-Spectro
SpecWdow=sps; % Length of th windos used to compute the spectrogram (samples)
noverlap=0.90*SpecWdow; % overlap of the moving window
SPEC{ii}=filter(ones(1,100)./100,1,(abs(spectrogram(SIG{ii},SpecWdow,noverlap,Freq1,sps))),[],1); % Compute and smooth Spectro
[SpecMaxEnv{ii},SpecMaxFreq{ii}]=max(SPEC{ii},[],1); % Max of each DFT in Spec
SpecMeanEnv{ii}=mean(SPEC{ii},1); % Mean of each DFT in Spec
SpecMedianEnv{ii}=median(SPEC{ii},1); % Median of each DFT in Spec
SpecVarEnv{ii}=var(SPEC{ii},1); % Var of each DFT in Spec
Xcentroid=sum(repmat(1:size(SPEC{ii},1),size(SPEC{ii},2),1)'.*SPEC{ii},1)./...
    sum(SPEC{ii},1); % Compute centroid of each DFT in Spec
for kk=1:round(size(SPEC{ii},2))
Xquart1(kk)=sum((1:length(SPEC{ii}(1:round(Xcentroid(kk)),kk)))'.*SPEC{ii}(1:round(Xcentroid(kk)),kk),1)./...
    sum(SPEC{ii}(1:round(Xcentroid(kk)),kk)); % Compute Q1 of each DFT in Spec
end
for kk2=1:round(size(SPEC{ii},2))
Xquart3(kk2)=round(Xcentroid(kk2))+sum((1:length(SPEC{ii}(round(Xcentroid(kk2)):end,kk2)))'...
    .*SPEC{ii}(round(Xcentroid(kk2)):end,kk2),1)./sum(SPEC{ii}(round(Xcentroid(kk2)):end,kk2)); % Compute Q3 of each DFT in Spec
end
FREQCENTER{ii}=Freq1(round(Xcentroid)); % Curve of the centroid position over time 
FREQQ1{ii}=Freq1(round(Xquart1));
FREQQ3{ii}=Freq1(round(Xquart3));
FREQMAXCENTER{ii}=Freq1(round(SpecMaxFreq{ii}));% Position of the maximum of DFT

% Transform to single value attributes
SpecKurtoMaxEnv(ii)=kurtosis(SpecMaxEnv{ii}); % Kurto of Spectrum Max env in Spec
SpecKurtoMedianEnv(ii)=kurtosis(SpecMedianEnv{ii}); % Kurto of Spectrum Median env in Spec
RATIOENVSPECMAXMEAN(ii)=mean(SpecMaxEnv{ii}./SpecMeanEnv{ii}); % Ratio Max DFT(t)/ Mean DFT(t)
RATIOENVSPECMAXMEDIAN(ii)=mean(SpecMaxEnv{ii}./SpecMedianEnv{ii});% Ratio Max DFT/ Median DFT
DISTMAXMEAN(ii)=mean(abs(SpecMaxEnv{ii}-SpecMeanEnv{ii})); % Mean distance bewteen Max DFT(t) Mean DFT(t)
DISTMAXMEDIAN(ii)=mean(abs(SpecMaxEnv{ii}-SpecMedianEnv{ii}));% Mean distance bewteen Max DFT Median DFT
[foo,ipeaksMAX{ii}]=findpeaks(SpecMaxEnv{ii}./max(SpecMaxEnv{ii}),'minpeakheight',0.75);
NBRPEAKMAX(ii)=length(ipeaksMAX{ii});% Nbr peaks Max DFTs
[foo,ipeaksMEAN{ii}]=findpeaks(SpecMeanEnv{ii}./max(SpecMeanEnv{ii}),'minpeakheight',0.75);
NBRPEAKMEAN(ii)=length(ipeaksMEAN{ii});% Nbr peaks Mean DFTs
[foo,ipeaksMEDIAN{ii}]=findpeaks(SpecMedianEnv{ii}./max(SpecMedianEnv{ii}),'minpeakheight',0.75);
NBRPEAKMEDIAN(ii)=length(ipeaksMEDIAN{ii});% Nbr peaks Median DFTs
RATIONBRPEAKMAXMEAN(ii)=NBRPEAKMAX(ii)/NBRPEAKMEAN(ii);% Ratio Max/Mean DFTs
RATIONBRPEAKMAXMED(ii)=NBRPEAKMAX(ii)/NBRPEAKMEDIAN(ii);% Ratio Max/Median DFTs
[foo,ipeaksFREQCENTER{ii}]=findpeaks(FREQCENTER{ii}./max(FREQCENTER{ii}),'minpeakheight',0.75);
NBRPEAKFREQCENTER(ii)=length(ipeaksFREQCENTER{ii});% Nbr peaks X centroid Freq DFTs
[foo,ipeaksFREQMAX{ii}]=findpeaks(FREQMAXCENTER{ii}./max(FREQMAXCENTER{ii}),'minpeakheight',0.75);
NBRPEAKFREQMAX(ii)=length(ipeaksFREQMAX{ii});% Nbr peaks X Max Freq DFTs
RATIONBRFREQPEAKS(ii)=NBRPEAKFREQMAX(ii)/NBRPEAKFREQCENTER(ii);% Ration Freq Max/X Centroid DFTs
DISTQ2Q1(ii)=mean(abs(FREQCENTER{ii}-FREQQ1{ii})); % Distance Q2 curve to Q1 curve
DISTQ3Q2(ii)=mean(abs(FREQQ3{ii}-FREQCENTER{ii})); % Distance Q3 curve to Q2 curve
DISTQ3Q1(ii)=mean(abs(FREQQ3{ii}-FREQQ1{ii})); % Distance Q3 curve to Q1 curve

end

%% Polarisation features

EndWindow=round(ImaxT2/3); % Window onto which compute the polarisation parameters (Default AScending time)
% A TESTer PLUSIEURS TAILLE FENETRE

if flag==2 
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
elseif flag==4
    cc=1;
    for ll=1:3:length(SIG)
            xP=SIG{ll+2}(1:EndWindow)';% Horiz 1
            yP=SIG{ll+1}(1:EndWindow)';% Horiz 2
            zP=SIG{ll}(1:EndWindow)';% Vertical (Up)
            MP=cov([xP yP zP]);%covariance(xP,yP,zP);
            [pP,DP] = eig(MP);
            rectilinP(cc)=1-((DP(1,1)+DP(2,2))/(2*DP(3,3)));
            azimuthP(cc)=atan(pP(2,3)/pP(1,3))*180/pi;
            dipP(cc)=atan(pP(3,3)/sqrt(pP(2,3)^2+pP(1,3)^2))*180/pi;
            Plani(cc)=1-(2*DP(1,1))/((DP(3,3)+DP(2,2)));
            cc=cc+1;
    end
else
end
%% Multi-station features

if flag==3
end

%% Misc features

%% SAVE IN ATTRIBUTES ARRAY

ATTRIBUTES=zeros(1,61);
ATTRIBUTESstd=zeros(1,57);

%-- Waveform
ATTRIBUTES(1)=mean(Duration);% Duration
ATTRIBUTES(2)=mean(RappMaxMean); % Ratio max/mean envelope
ATTRIBUTES(3)=mean(RappMaxMedian); % Ratio max/median envelope
ATTRIBUTES(4)=mean(TASSENCDES); % Ascending time/Decreasing time of the envelope
ATTRIBUTES(5)=mean(KurtoSig); % Kurtosis Signal
ATTRIBUTES(6)=mean(KurtoEnv); % Kurtosis Envelope
ATTRIBUTES(7)=mean(abs(SkewnessSig)); % Skewness Signal
ATTRIBUTES(8)=mean(abs(SkewnessEnv)); % Skewness envelope
ATTRIBUTES(9)=mean(CORPEAKNUMBER(ii)); % Nombre of peaks in the autocorrelation function
ATTRIBUTES(10)=mean(INT1); % Energy in the 1/3 of the autocorr function
ATTRIBUTES(11)=mean(INT2); % Energy in the last 2/3 of the autocorr function
ATTRIBUTES(12)=mean(INT); % Ration of the aboves energies
ATTRIBUTES(13)=mean(ES(:,1)); % Energy of the seismic signal in the 0.1-1Hz FBand
ATTRIBUTES(14)=mean(ES(:,2)); % Energy of the seismic signal in the 1-3Hz FBand
ATTRIBUTES(15)=mean(ES(:,3)); % Energy of the seismic signal in the 3-10Hz FBand
ATTRIBUTES(16)=mean(ES(:,4)); % Energy of the seismic signal in the 10-20Hz FBand
ATTRIBUTES(17)=mean(ES(:,5)); % Energy of the seismic signal in the 20-Nyquist F FBand
ATTRIBUTES(18)=mean(KurtoF(:,1)); % Kurtosis of the signal in the 0.1-1Hz FBand
ATTRIBUTES(19)=mean(KurtoF(:,2)); % Kurtosis of the signal in the 1-3Hz FBand
ATTRIBUTES(20)=mean(KurtoF(:,3)); % Kurtosis of the signal in the 3-10Hz FBand
ATTRIBUTES(21)=mean(KurtoF(:,4)); % Kurtosis of the signal in the 10-20Hz FBand
ATTRIBUTES(22)=mean(KurtoF(:,5)); % Kurtosis of the signal in the 20-Nyf Hz FBand
ATTRIBUTES(23)=mean(RMSDECAMPENV); % RMS Between amplitude decreasing amplitude and straight line

%-- Spectral 

%Full Spectrum
ATTRIBUTES(24)=mean(MEANFFT); % Mean FFT
ATTRIBUTES(25)=mean(MAXFFT); % Max FFT
ATTRIBUTES(26)=mean(FMAX); % Frequence at Max(FFT)
ATTRIBUTES(27)=mean(FCENTROID); % Frequence centroid
ATTRIBUTES(28)=mean(FQUART1); %Frequence 1 quartile
ATTRIBUTES(29)=mean(FQUART3); %Frequence 3 quartile
ATTRIBUTES(30)=mean(MEDIANFFT); % Median Normalized FFT
ATTRIBUTES(31)=mean(VARFFT); % var Normalized FFT
ATTRIBUTES(32)=mean(NBRPEAKFFT); % Number of peaks in normalized fft
ATTRIBUTES(33)=mean(MEANPEAKS); % Mean peaks value for peaks>0.7
ATTRIBUTES(34)=mean(E1FFT); %Energy in the 1-12.5Hz 
ATTRIBUTES(35)=mean(E2FFT); %idem 12.5-25Hz
ATTRIBUTES(36)=mean(E3FFT); %idem 25-37.5Hz
ATTRIBUTES(37)=mean(E4FFT); % idem 37.5-50Hz
ATTRIBUTES(38)=mean(gamma1); % spectral centroid
ATTRIBUTES(39)=mean(gamma2); % spectral gyration radius
ATTRIBUTES(40)=mean(gammas); % spectral centroid width

%Pseudo-Spectrogram
ATTRIBUTES(41)=mean(SpecKurtoMaxEnv); % Kurto of Spectrum Max env in Spec
ATTRIBUTES(42)=mean(SpecKurtoMedianEnv); % Kurto of Spectrum Median env in Spec
ATTRIBUTES(43)=mean(RATIOENVSPECMAXMEAN); % Ratio Max DFT(t)/ Mean DFT(t)
ATTRIBUTES(44)=mean(RATIOENVSPECMAXMEDIAN); % Ratio Max DFT/ Median DFT
ATTRIBUTES(45)=mean(NBRPEAKMAX); % Nbr peaks Max DFTs
ATTRIBUTES(46)=mean(NBRPEAKMEAN); % Nbr peaks Mean DFTs
ATTRIBUTES(47)=mean(NBRPEAKMEDIAN); % Nbr peaks Median DFTs
ATTRIBUTES(48)=mean(RATIONBRPEAKMAXMEAN); % Ratio Max/Mean DFTs
ATTRIBUTES(49)=mean(RATIONBRPEAKMAXMED); % Ratio Max/Median DFTs
ATTRIBUTES(50)=mean(NBRPEAKFREQCENTER); % Nbr peaks X centroid Freq DFTs
ATTRIBUTES(51)=mean(NBRPEAKFREQMAX); % Nbr peaks X Max Freq DFTs
ATTRIBUTES(52)=mean(RATIONBRFREQPEAKS); % Ration Freq Max/X Centroid DFTs
ATTRIBUTES(53)=mean(DISTMAXMEAN); % Mean distance bewteen Max DFT(t) Mean DFT(t)
ATTRIBUTES(54)=mean(DISTMAXMEDIAN);% Mean distance bewteen Max DFT Median DFT
ATTRIBUTES(55)=mean(DISTQ2Q1); % Distance Q2 curve to Q1 curve
ATTRIBUTES(56)=mean(DISTQ3Q2); % Distance Q3 curve to Q2 curve
ATTRIBUTES(57)=mean(DISTQ3Q1); % Distance Q3 curve to Q1 curve

%-- Polarisation

if flag==2
    % Polarisation Parameter
            ATTRIBUTES(58)=rectilinP;% Rectinilarity
            ATTRIBUTES(59)=azimuthP;  % Azimuth - Not Necessary?
            ATTRIBUTES(60)=dipP; % Dip
            ATTRIBUTES(61)=Plani; % Planarity
            
    % STD of ATTRIBUTES COMPUTATION
    %-- Waveform
            ATTRIBUTESstd(1)=std(Duration);% Duration
            ATTRIBUTESstd(2)=std(RappMaxMean); % Ratio max/std envelope
            ATTRIBUTESstd(3)=std(RappMaxMedian); % Ratio max/median envelope
            ATTRIBUTESstd(4)=std(TASSENCDES); % Ascending time/Decreasing time of the envelope
            ATTRIBUTESstd(5)=std(KurtoSig); % Kurtosis Signal
            ATTRIBUTESstd(6)=std(KurtoEnv); % Kurtosis Envelope
            ATTRIBUTESstd(7)=std(SkewnessSig); % Skewness Signal
            ATTRIBUTESstd(8)=std(SkewnessEnv); % Skewness envelope
            ATTRIBUTESstd(9)=std(CORPEAKNUMBER(ii)); % Nombre of peaks in the autocorrelation function
            ATTRIBUTESstd(10)=std(INT1); % Energy in the 1/3 of the autocorr function
            ATTRIBUTESstd(11)=std(INT2); % Energy in the last 2/3 of the autocorr function
            ATTRIBUTESstd(12)=std(INT); % Ration of the aboves energies
            ATTRIBUTESstd(13)=std(ES(:,1)); % Energy of the seismic signal in the 0.1-1Hz FBand
            ATTRIBUTESstd(14)=std(ES(:,2)); % Energy of the seismic signal in the 1-3Hz FBand
            ATTRIBUTESstd(15)=std(ES(:,3)); % Energy of the seismic signal in the 3-10Hz FBand
            ATTRIBUTESstd(16)=std(ES(:,4)); % Energy of the seismic signal in the 10-20Hz FBand
            ATTRIBUTESstd(17)=std(ES(:,5)); % Energy of the seismic signal in the 20-Nyquist F FBand
            ATTRIBUTESstd(18)=std(KurtoF(:,1)); % Kurtosis of the signal in the 0.1-1Hz FBand
            ATTRIBUTESstd(19)=std(KurtoF(:,2)); % Kurtosis of the signal in the 1-3Hz FBand
            ATTRIBUTESstd(20)=std(KurtoF(:,3)); % Kurtosis of the signal in the 3-10Hz FBand
            ATTRIBUTESstd(21)=std(KurtoF(:,4)); % Kurtosis of the signal in the 10-20Hz FBand
            ATTRIBUTESstd(22)=std(KurtoF(:,5)); % Kurtosis of the signal in the 20-Nyf Hz FBand
            ATTRIBUTESstd(23)=std(RMSDECAMPENV); % RMS Between amplitude decreasing amplitude and straight line

            %-- Spectral 

            %Full Spectrum
            ATTRIBUTESstd(24)=std(MEANFFT); % Mean FFT
            ATTRIBUTESstd(25)=std(MAXFFT); % Max FFT
            ATTRIBUTESstd(26)=std(FMAX); % Frequence at Max(FFT)
            ATTRIBUTESstd(27)=std(FCENTROID); % Frequence centroid
            ATTRIBUTESstd(28)=std(FQUART1); %Frequence 1 quartile
            ATTRIBUTESstd(29)=std(FQUART3); %Frequence 3 quartile
            ATTRIBUTESstd(30)=std(MEDIANFFT); % Median Normalized FFT
            ATTRIBUTESstd(31)=std(VARFFT); % var Normalized FFT
            ATTRIBUTESstd(32)=std(NBRPEAKFFT); % Number of peaks in normalized fft
            ATTRIBUTESstd(33)=std(MEANPEAKS); % Mean peaks value for peaks>0.7
            ATTRIBUTESstd(34)=std(E1FFT); %Energy in the 1-12.5Hz 
            ATTRIBUTESstd(35)=std(E2FFT); %idem 12.5-25Hz
            ATTRIBUTESstd(36)=std(E3FFT); %idem 25-37.5Hz
            ATTRIBUTESstd(37)=std(E4FFT); % idem 37.5-50Hz
            ATTRIBUTESstd(38)=std(gamma1); % spectral centroid
            ATTRIBUTESstd(39)=std(gamma2); % spectral gyration radius
            ATTRIBUTESstd(40)=std(gammas); % spectral centroid width


            %Pseudo-Spectorgram
            ATTRIBUTESstd(41)=std(SpecKurtoMaxEnv); % Kurto of Spectrum Max env in Spec
            ATTRIBUTESstd(42)=std(SpecKurtoMedianEnv); % Kurto of Spectrum Median env in Spec
            ATTRIBUTESstd(43)=std(RATIOENVSPECMAXMEAN); % Ratio Max DFT(t)/ Mean DFT(t)
            ATTRIBUTESstd(44)=std(RATIOENVSPECMAXMEDIAN);% Ratio Max DFT/ Median DFT
            ATTRIBUTESstd(45)=std(NBRPEAKMAX);% Nbr peaks Max DFTs
            ATTRIBUTESstd(46)=std(NBRPEAKMEAN);% Nbr peaks Mean DFTs
            ATTRIBUTESstd(47)=std(NBRPEAKMEDIAN);% Nbr peaks Median DFTs
            ATTRIBUTESstd(48)=std(RATIONBRPEAKMAXMEAN);% Ratio Max/Mean DFTs
            ATTRIBUTESstd(49)=std(RATIONBRPEAKMAXMED);% Ratio Max/Median DFTs
            ATTRIBUTESstd(50)=std(NBRPEAKFREQCENTER);% Nbr peaks X centroid Freq DFTs
            ATTRIBUTESstd(51)=std(NBRPEAKFREQMAX);% Nbr peaks X Max Freq DFTs
            ATTRIBUTESstd(52)=std(RATIONBRFREQPEAKS);% Ration Freq Max/X Centroid DFTs
            ATTRIBUTESstd(53)=std(DISTMAXMEAN); % Mean distance bewteen Max DFT(t) Mean DFT(t)
            ATTRIBUTESstd(54)=std(DISTMAXMEDIAN);% Mean distance bewteen Max DFT Median DFT
            ATTRIBUTESstd(55)=std(DISTQ2Q1); % Distance Q2 curve to Q1 curve
            ATTRIBUTESstd(56)=std(DISTQ3Q2); % Distance Q3 curve to Q2 curve
            ATTRIBUTESstd(57)=std(DISTQ3Q1); % Distance Q3 curve to Q1 curve
else
    
end

%-- MultiStation
%Coming Later
end
