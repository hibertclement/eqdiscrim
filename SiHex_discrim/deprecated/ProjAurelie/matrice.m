clear all 
close all
clc

tic
file=fopen('SIHEXV2-catalogue-final.txt');
seisme=textscan(file,'%f %s %s %f %f %f %s %s %f','HeaderLines',4);
fclose(file);
file=fopen('no_tecto.lst');
notecto=textscan(file,'%f %s %s %f %f %f %s %s %f');
fclose(file);

NbrSeismes=size(seisme{1},1);  % nombre de seismes total
NbrSeismes1=size(notecto{1},1);  %nombre de non tecto total

%%%% creation du vecteur du type de seisme : ke,...
V1=seisme{8};
V2=notecto{8};
Vtot=[V1;V2];

%%%%%%%%%%%%%%%%%%%%%%%%%%%% colonnes de la matrice %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%colonne1 : numero de l'evenement
Mke(:,1)=seisme{1};
Mkm(:,1)=notecto{1};
M1=[Mke;Mkm];

%%%%%%%%%%%%%%%%%%%%%colonne2 : jour de la semaine
%%% pour les seismes
 Dateheure = strcat(seisme{2},{' '},seisme{3});
 dateheureA= datenum(Dateheure,'yyyy/mm/dd HH:MM:SS',1962);
 datecompvec=datevec(dateheureA);
% les dates localement
datelocale= utc_to_local_time( datecompvec ); 
%transformation des dates locales en nombres lisibles par matlab
C=[];
for i=1:NbrSeismes %separation des dates et des heures en deux colonnes
dateheure=strsplit(datelocale(i,:),' ');
C= [C;dateheure(1),dateheure(2)]; %C est un tableau 2*X avec les dates et les heures
end 
%%recuperation des jours de la semaine
numjourok = weekday (C(:,1));

%%% pour les non tecto
 Dateheure1 = strcat(notecto{2},{' '},notecto{3});
 dateheureA1= datenum(Dateheure1,'yyyy/mm/dd HH:MM:SS',1967);
 datecompvec1=datevec(dateheureA1);
% les dates localement
datelocale1= utc_to_local_time( datecompvec1 ); 
%transformation des dates locales en nombres lisibles par matlab
C1=[];
for i=1:NbrSeismes1 %separation des dates et des heures en deux colonnes
dateheure1=strsplit(datelocale1(i,:),' ');
C1= [C1;dateheure1(1),dateheure1(2)]; %C est un tableau 2*X avec les dates et les heures
end 
%%recuperation des jours de la semaine
numjourok1 = weekday (C1(:,1));
M2=[numjourok; numjourok1];

%%%%%%%%%%%%%%%% colonne 3 : Latitude
M3ke=seisme{4};
M3km=notecto{4};
M3=[M3ke;M3km];

%%%%%%%%%%%%%%% colonne 4 : Longitude
M4ke=seisme{5};
M4km=notecto{5};
M4=[M4ke;M4km];

%%%%%%%%%%%%%%% colonne 5 : heure de la journée
%%% pour les seismes
heurehist1 = datevec(C(:,2));
heurehist=heurehist1(:,4); 
%%% pour les non tectos
heurehist2 = datevec(C1(:,2));
heurehistkm=heurehist2(:,4);

M5=[heurehist;heurehistkm];

Mat = [M1 M2 M3 M4];  %%% matrice finale !
toc