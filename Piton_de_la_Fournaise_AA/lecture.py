# -*- coding: utf-8 -*-

from do_class_test import r_file, stat_mean, stat_variance, stat_standard_deviation
from matplotlib import pyplot as plt
import numpy as np

######################################## IMPORTANT VALUES

# value for each station
#cv
val1, key1 = r_file('CVdict01.dat')
val2, key2 = r_file('CVdict02.dat')
val3, key3 = r_file('CVdict11.dat')
val4, key4 = r_file('CVdict12.dat')
val5, key5 = r_file('CVdict21.dat')
val6, key6 = r_file('CVdict22.dat')
val7, key7 = r_file('CVdict31.dat')
val8, key8 = r_file('CVdict32.dat')

print val1[6]
print val2[6]

#ar
arv1, ark1 = r_file('ARdict01.dat')
arv2, ark2 = r_file('ARdict02.dat')
arv3, ark3 = r_file('ARdict11.dat')
arv4, ark4 = r_file('ARdict12.dat')
arv5, ark5 = r_file('ARdict21.dat')
arv6, ark6 = r_file('ARdict22.dat')
arv7, ark7 = r_file('ARdict31.dat')
arv8, ark8 = r_file('ARdict32.dat')
#cvm
valm1, keym1 = r_file('CVMdict01.dat')
valm2, keym2 = r_file('CVMdict02.dat')
valm3, keym3 = r_file('CVMdict11.dat')
valm4, keym4 = r_file('CVMdict12.dat')
valm5, keym5 = r_file('CVMdict21.dat')
valm6, keym6 = r_file('CVMdict22.dat')
valm7, keym7 = r_file('CVMdict31.dat')
valm8, keym8 = r_file('CVMdict32.dat')

# mean and standard deviation
# IN PROGRESS




# max cv per combination for each station

maxsta1 = []
maxsta2 = []
maxsta3 = []
maxsta4 = []
maxsta5 = []
maxsta6 = []
maxsta7 = []
maxsta8 = []


print 'RVL'
max011 = np.max(val1[0]); print max011; maxsta1.append(max011)
max012 = np.max(val1[1]); print max012; maxsta1.append(max012)
max013 = np.max(val1[2]); print max013; maxsta1.append(max013)
max014 = np.max(val1[3]); print max014; maxsta1.append(max014)
max015 = np.max(val1[4]); print max015; maxsta1.append(max015)
max016 = np.max(val1[5]); print max016; maxsta1.append(max016)
max017 = np.max(val1[6]); print max017; maxsta1.append(max017)
max018 = np.max(val1[7]); print max018; maxsta1.append(max018)

print 'FJS'
max021 = np.max(val2[0]); print max021; maxsta2.append(max021)
max022 = np.max(val2[1]); print max022; maxsta2.append(max022)
max023 = np.max(val2[2]); print max023; maxsta2.append(max023)
max024 = np.max(val2[3]); print max024; maxsta2.append(max024)
max025 = np.max(val2[4]); print max025; maxsta2.append(max025)
max026 = np.max(val2[5]); print max026; maxsta2.append(max026)
max027 = np.max(val2[6]); print max027; maxsta2.append(max027)
max028 = np.max(val2[7]); print max028; maxsta2.append(max028)

print 'GPS'
max111 = np.max(val3[0]); print max111; maxsta3.append(max111)
max112 = np.max(val3[1]); print max112; maxsta3.append(max112)
max113 = np.max(val3[2]); print max113; maxsta3.append(max113)
max114 = np.max(val3[3]); print max114; maxsta3.append(max114)
max115 = np.max(val3[4]); print max115; maxsta3.append(max115)
max116 = np.max(val3[5]); print max116; maxsta3.append(max116)
max117 = np.max(val3[6]); print max117; maxsta3.append(max117)
max118 = np.max(val3[7]); print max118; maxsta3.append(max118)

print 'SNE'
max121 = np.max(val4[0]); print max121; maxsta4.append(max121)
max122 = np.max(val4[1]); print max122; maxsta4.append(max122)
max123 = np.max(val4[2]); print max123; maxsta4.append(max123)
max124 = np.max(val4[3]); print max124; maxsta4.append(max124)
max125 = np.max(val4[4]); print max125; maxsta4.append(max125)
max126 = np.max(val4[5]); print max126; maxsta4.append(max126)
max127 = np.max(val4[6]); print max127; maxsta4.append(max127)
max128 = np.max(val4[7]); print max128; maxsta4.append(max128)

print 'BOR'
max211 = np.max(val5[0]); print max211; maxsta5.append(max211)
max212 = np.max(val5[1]); print max212; maxsta5.append(max212)
max213 = np.max(val5[2]); print max213; maxsta5.append(max213)
max214 = np.max(val5[3]); print max214; maxsta5.append(max214)
max215 = np.max(val5[4]); print max215; maxsta5.append(max215)
max216 = np.max(val5[5]); print max216; maxsta5.append(max216)
max217 = np.max(val5[6]); print max217; maxsta5.append(max217)
max218 = np.max(val5[7]); print max218; maxsta5.append(max218)

print 'DSO'
max221 = np.max(val6[0]); print max221; maxsta6.append(max221)
max222 = np.max(val6[1]); print max222; maxsta6.append(max222)
max223 = np.max(val6[2]); print max223; maxsta6.append(max223)
max224 = np.max(val6[3]); print max224; maxsta6.append(max224)
max225 = np.max(val6[4]); print max225; maxsta6.append(max225)
max226 = np.max(val6[5]); print max226; maxsta6.append(max226)
max227 = np.max(val6[6]); print max227; maxsta6.append(max227)
max228 = np.max(val6[7]); print max228; maxsta6.append(max228)

print'ENO'
max311 = np.max(val7[0]); print max311; maxsta7.append(max311)
max312 = np.max(val7[1]); print max312; maxsta7.append(max312)
max313 = np.max(val7[2]); print max313; maxsta7.append(max313)
max314 = np.max(val7[3]); print max314; maxsta7.append(max314)
max315 = np.max(val7[4]); print max315; maxsta7.append(max315)
max316 = np.max(val7[5]); print max316; maxsta7.append(max316)
max317 = np.max(val7[6]); print max317; maxsta7.append(max317)
max318 = np.max(val7[7]); print max318; maxsta7.append(max318)

print 'PHR'
max321 = np.max(val8[0]); print max321; maxsta8.append(max321)
max322 = np.max(val8[1]); print max322; maxsta8.append(max322)
max323 = np.max(val8[2]); print max323; maxsta8.append(max323)
max324 = np.max(val8[3]); print max324; maxsta8.append(max324)
max325 = np.max(val8[4]); print max325; maxsta8.append(max325)
max326 = np.max(val8[5]); print max326; maxsta8.append(max326)
max327 = np.max(val8[6]); print max327; maxsta8.append(max327)
max328 = np.max(val8[7]); print max328; maxsta8.append(max328)

# max per station

MAX = []
MAXSTA1 = np.max(maxsta1); MAX.append(MAXSTA1)
MAXSTA2 = np.max(maxsta2); MAX.append(MAXSTA2)
MAXSTA3 = np.max(maxsta3); MAX.append(MAXSTA3)
MAXSTA4 = np.max(maxsta4); MAX.append(MAXSTA4)

MAXSTA5 = np.max(maxsta5); MAX.append(MAXSTA5)
MAXSTA6 = np.max(maxsta6); MAX.append(MAXSTA6)
MAXSTA7 = np.max(maxsta7); MAX.append(MAXSTA7)
MAXSTA8 = np.max(maxsta8); MAX.append(MAXSTA8)

print 'the cv max of all value is %.4f'%(np.max(MAX))


# max ar 

maxsta1 = []
maxsta2 = []
maxsta3 = []
maxsta4 = []
maxsta5 = []
maxsta6 = []
maxsta7 = []
maxsta8 = []


print 'RVL'
max011 = np.max(arv1[0]); print max011; maxsta1.append(max011)
max012 = np.max(arv1[1]); print max012; maxsta1.append(max012)
max013 = np.max(arv1[2]); print max013; maxsta1.append(max013)
max014 = np.max(arv1[3]); print max014; maxsta1.append(max013)
max015 = np.max(arv1[4]); print max015; maxsta1.append(max014)
max016 = np.max(arv1[5]); print max016; maxsta1.append(max015)
max017 = np.max(arv1[6]); print max017; maxsta1.append(max016)
max018 = np.max(arv1[7]); print max018; maxsta1.append(max017)

print 'FJS'
max021 = np.max(arv2[0]); print max021; maxsta2.append(max021)
max022 = np.max(arv2[1]); print max022; maxsta2.append(max022)
max023 = np.max(arv2[2]); print max023; maxsta2.append(max023)
max024 = np.max(arv2[3]); print max024; maxsta2.append(max024)
max025 = np.max(arv2[4]); print max025; maxsta2.append(max025)
max026 = np.max(arv2[5]); print max026; maxsta2.append(max026)
max027 = np.max(arv2[6]); print max027; maxsta2.append(max027)
max028 = np.max(arv2[7]); print max028; maxsta2.append(max027)

print 'GPS'
max111 = np.max(arv3[0]); print max111; maxsta3.append(max111)
max112 = np.max(arv3[1]); print max112; maxsta3.append(max112)
max113 = np.max(arv3[2]); print max113; maxsta3.append(max113)
max114 = np.max(arv3[3]); print max114; maxsta3.append(max114)
max115 = np.max(arv3[4]); print max115; maxsta3.append(max115)
max116 = np.max(arv3[5]); print max116; maxsta3.append(max116)
max117 = np.max(arv3[6]); print max117; maxsta3.append(max117)
max118 = np.max(arv3[7]); print max118; maxsta3.append(max118)

print 'SNE'
max121 = np.max(arv4[0]); print max121; maxsta4.append(max121)
max122 = np.max(arv4[1]); print max122; maxsta4.append(max122)
max123 = np.max(arv4[2]); print max123; maxsta4.append(max123)
max124 = np.max(arv4[3]); print max124; maxsta4.append(max124)
max125 = np.max(arv4[4]); print max125; maxsta4.append(max125)
max126 = np.max(arv4[5]); print max126; maxsta4.append(max126)
max127 = np.max(arv4[6]); print max127; maxsta4.append(max127)
max128 = np.max(arv4[7]); print max128; maxsta4.append(max128)

print 'BOR'
max211 = np.max(arv5[0]); print max211; maxsta5.append(max211)
max212 = np.max(arv5[1]); print max212; maxsta5.append(max212)
max213 = np.max(arv5[2]); print max213; maxsta5.append(max213)
max214 = np.max(arv5[3]); print max214; maxsta5.append(max214)
max215 = np.max(arv5[4]); print max215; maxsta5.append(max215)
max216 = np.max(arv5[5]); print max216; maxsta5.append(max216)
max217 = np.max(arv5[6]); print max217; maxsta5.append(max217)
max218 = np.max(arv5[7]); print max218; maxsta5.append(max218)

print 'DSO'
max221 = np.max(arv6[0]); print max221; maxsta6.append(max221)
max222 = np.max(arv6[1]); print max222; maxsta6.append(max222)
max223 = np.max(arv6[2]); print max223; maxsta6.append(max223)
max224 = np.max(arv6[3]); print max224; maxsta6.append(max224)
max225 = np.max(arv6[4]); print max225; maxsta6.append(max225)
max226 = np.max(arv6[5]); print max226; maxsta6.append(max226)
max227 = np.max(arv6[6]); print max227; maxsta6.append(max227)
max228 = np.max(arv6[7]); print max228; maxsta6.append(max228)

print'ENO'
max311 = np.max(arv7[0]); print max311; maxsta7.append(max311)
max312 = np.max(arv7[1]); print max312; maxsta7.append(max312)
max313 = np.max(arv7[2]); print max313; maxsta7.append(max313)
max314 = np.max(arv7[3]); print max314; maxsta7.append(max314)
max315 = np.max(arv7[4]); print max315; maxsta7.append(max315)
max316 = np.max(arv7[5]); print max316; maxsta7.append(max316)
max317 = np.max(arv7[6]); print max317; maxsta7.append(max317)
max318 = np.max(arv7[7]); print max318; maxsta7.append(max318)

print 'PHR'
max321 = np.max(arv8[0]); print max321; maxsta8.append(max321)
max322 = np.max(arv8[1]); print max322; maxsta8.append(max322)
max323 = np.max(arv8[2]); print max323; maxsta8.append(max323)
max324 = np.max(arv8[3]); print max324; maxsta8.append(max324)
max325 = np.max(arv8[4]); print max325; maxsta8.append(max325)
max326 = np.max(arv8[5]); print max326; maxsta8.append(max326)
max327 = np.max(arv8[6]); print max327; maxsta8.append(max327)
max328 = np.max(arv8[7]); print max328; maxsta8.append(max328)


MAX2 = []
MAXSTA1 = np.max(maxsta1); MAX2.append(MAXSTA1)
MAXSTA2 = np.max(maxsta2); MAX2.append(MAXSTA2)
MAXSTA3 = np.max(maxsta3); MAX2.append(MAXSTA3)
MAXSTA4 = np.max(maxsta4); MAX2.append(MAXSTA4)

MAXSTA5 = np.max(maxsta5); MAX2.append(MAXSTA5)
MAXSTA6 = np.max(maxsta6); MAX2.append(MAXSTA6)
MAXSTA7 = np.max(maxsta7); MAX2.append(MAXSTA7)
MAXSTA8 = np.max(maxsta8); MAX2.append(MAXSTA8)

print 'the ar max value is %.4f'%(np.max(MAX2))

print 'the cv max of all value is %.4f'%(np.max(MAX))


######################################## PLOT



# CV
############ Combinations of one attribut

fig = plt.figure(figsize = (10,15))
fig.subplots_adjust(left = 0.06, bottom = 0.07,
                     right = 0.8, top = 0.96, wspace = 0.5, hspace = 0.5)

ax = plt.subplot(4,1,1)

plt.errorbar(range(len(val1[0])), val1[0], yerr=valm1[0], marker='.', color = 'y')
plt.errorbar(range(len(val1[0])), val2[0], yerr=valm2[0], marker='.', color = 'c')
plt.errorbar(range(len(val1[0])), val3[0], yerr=valm3[0], marker='.')
plt.errorbar(range(len(val1[0])), val4[0], yerr=valm4[0], marker='.', color = 'g')

plt.scatter(range(len(arv1[0])), arv1[0], marker='o', color = 'y', s= 80)
plt.scatter(range(len(arv1[0])), arv2[0], marker='o', color = 'c', s= 80)
plt.scatter(range(len(arv1[0])), arv3[0], marker='o', color = 'b', s= 80)
plt.scatter(range(len(arv1[0])), arv4[0], marker='o', color = 'g', s= 80)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

# Put a legend to the right of the current axis
ax.legend(['RVL', 'FJS', 'GPS', 'SNE', 'RVL_ar', 'FJS_ar', 'GPS_ar', 'SNE_ar'], loc='center left', bbox_to_anchor=(1, 0.5))        
ax.text(-0.6, 0.31, 'maxAmean', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
ax.text(0.9, 0.31, 'Dur/A', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
ax.text(1.9, 0.31, 'K', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
ax.text(2.9, 0.31, 'Dur', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
ax.text(3.9, 0.31, 'A', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
ax.text(4.9, 0.31, 'E', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
ax.text(5.9, 0.31, 'dom_f', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
ax.text(6.9, 0.31, 'cent_f', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
        
plt.axis()
plt.title('cv/cvm_LB_1')
plt.ylim(0.3,1)

ax = plt.subplot(4,1,2)

plt.errorbar(range(len(val5[0])), val5[0], yerr=valm5[0], marker='.', color = 'black')
plt.errorbar(range(len(val5[0])), val6[0], yerr=valm6[0], marker='.', color = 'c')
plt.errorbar(range(len(val5[0])), val7[0], yerr=valm7[0], marker='.', color = 'r')
plt.errorbar(range(len(val5[0])), val8[0], yerr=valm8[0], marker='.', color = 'g')

plt.scatter(range(len(arv5[0])), arv5[0], marker='o', color = 'black', s= 80)
plt.scatter(range(len(arv5[0])), arv6[0], marker='o', color = 'c', s= 80)
plt.scatter(range(len(arv5[0])), arv7[0], marker='o', color = 'r', s= 80)
plt.scatter(range(len(arv5[0])), arv8[0], marker='o', color = 'g', s= 80)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

# Put a legend to the right of the current axis
ax.legend(['BOR', 'DSO', 'ENO', 'PHR', 'BOR_ar', 'DOS_ar', 'ENO_ar', 'PHR_ar'], loc='center left', bbox_to_anchor=(1, 0.5))
ax.text(-0.6, 0.31, 'maxAmean', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
ax.text(0.9, 0.31, 'Dur/A', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
ax.text(1.9, 0.31, 'K', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
ax.text(2.9, 0.31, 'Dur', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
ax.text(3.9, 0.31, 'A', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
ax.text(4.9, 0.31, 'E', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
ax.text(5.9, 0.31, 'dom_f', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
ax.text(6.9, 0.31, 'cent_f', style='normal',fontsize = 13,
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})
              
plt.axis()
plt.title('cv/cvm_CP_1')
plt.ylim(0.3,1)

################ combination of Two attributs

ax = plt.subplot(4,1,3)
plt.errorbar(range(len(val1[1])), val1[1], yerr=valm1[1], marker='.', color = 'y')
plt.errorbar(range(len(val1[1])), val2[1], yerr=valm2[1], marker='.', color = 'c')
plt.errorbar(range(len(val1[1])), val3[1], yerr=valm3[1], marker='.')
plt.errorbar(range(len(val1[1])), val4[1], yerr=valm4[1], marker='.', color = 'g')

ax.annotate('K-A-domf', xy=(3, 0.45), xytext=(3, 0.28),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('DurA-domf-maxAmean', xy=(9, 0.57), xytext=(9, 0.21),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('DurA-A-maxAmean', xy=(15, 0.47), xytext=(15, 0.26),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('K-E-maxAmean', xy=(17, 0.68), xytext=(17, 0.35),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('centf-Dur-DurA', xy=(21, 0.50), xytext=(21, 0.41),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.axis()
plt.title('cv/cvm_LB_2')
plt.ylim(0.2,1)
plt.xlim(-1, 28)

ax = plt.subplot(4,1,4)
plt.errorbar(range(len(val5[1])), val5[1], yerr=valm5[1], marker='.', color = 'black')
plt.errorbar(range(len(val5[1])), val6[1], yerr=valm6[1], marker='.', color = 'c')
plt.errorbar(range(len(val5[1])), val7[1], yerr=valm7[1], marker='.', color = 'r')
plt.errorbar(range(len(val5[1])), val8[1], yerr=valm8[1], marker='.', color = 'g')

ax.annotate('centf-Dur-A', xy=(0, 0.50), xytext=(0, 0.28),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('K-A-domf', xy=(3, 0.52), xytext=(3, 0.38),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('K-domf-E', xy=(8, 0.45), xytext=(8, 0.21),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('centf-K-domf', xy=(19, 0.55), xytext=(19, 0.31),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('K-E-maxAmean', xy=(17, 0.58), xytext=(17, 0.31),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('K-A-E', xy=(22, 0.54), xytext=(22, 0.35),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.axis()
plt.title('cv/cvm_CP_2')
plt.ylim(0.2,1)
plt.xlim(-1, 28)

fig.savefig('cv_cvm_1_2.png')
plt.clf()

######################## combinations of three attributs

fig = plt.figure(figsize = (10,15))
fig.subplots_adjust(left = 0.06, bottom = 0.07,
                     right = 0.8, top = 0.96, wspace = 0.5, hspace = 0.5)

ax = plt.subplot(4,1,1)

plt.errorbar(range(len(val1[2])), val1[2], yerr=valm1[2], marker='.', color = 'y')
plt.errorbar(range(len(val1[2])), val2[2], yerr=valm2[2], marker='.', color = 'c')
plt.errorbar(range(len(val1[2])), val3[2], yerr=valm3[2], marker='.')
plt.errorbar(range(len(val1[2])), val4[2], yerr=valm4[2], marker='.', color = 'g')


# Put a legend to the right of the current axis
ax.legend(['RVL', 'FJS', 'GPS', 'SNE'], loc='center left', bbox_to_anchor=(1, 0.5))

ax.annotate('K-A-domf', xy=(2, 0.52), xytext=(2, 0.38),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('DurA-domf-maxAmean', xy=(9, 0.62), xytext=(7, 0.31),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('DurA-A-maxAmean', xy=(20, 0.59), xytext=(20, 0.31),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('K-E-maxAmean', xy=(24, 0.55), xytext=(24, 0.37),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('centf-Dur-DurA', xy=(31, 0.59), xytext=(31, 0.45),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('centf-domf-maxAmean', xy=(40, 0.55), xytext=(40, 0.31),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('centf-K-A', xy=(47, 0.55), xytext=(47, 0.35),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('K-domf-maxAmean', xy=(50, 0.5), xytext=(50, 0.39),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.axis()
plt.title('cv/cvm_LB_3')
plt.ylim(0.3,1)
plt.xlim(-1, 56)


ax = plt.subplot(4,1,2)
plt.errorbar(range(len(val5[2])), val5[2], yerr=valm5[2], marker='.', color = 'black')
plt.errorbar(range(len(val5[2])), val6[2], yerr=valm6[2], marker='.', color = 'c')
plt.errorbar(range(len(val5[2])), val7[2], yerr=valm7[2], marker='.', color = 'r')
plt.errorbar(range(len(val5[2])), val8[2], yerr=valm8[2], marker='.', color = 'g')


# Put a legend to the right of the current axis
ax.legend(['BOR', 'DSO', 'ENO', 'PHR'], loc='center left', bbox_to_anchor=(1, 0.5))

ax.annotate('centf-Dur-A', xy=(0, 0.58), xytext=(0, 0.45), fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('K-A-domf', xy=(2, 0.65), xytext=(2, 0.49),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('DurA-domf-maxAmean', xy=(9, 0.49), xytext=(7, 0.31),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('DurA-A-maxAmean', xy=(20, 0.59), xytext=(20, 0.31),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Dur-K-maxAmean', xy=(25, 0.65), xytext=(25, 0.37),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('centf-Dur-DurA', xy=(31, 0.59), xytext=(31, 0.45),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('centf-domf-maxAmean', xy=(40, 0.55), xytext=(40, 0.35),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('centf-K-A', xy=(47, 0.55), xytext=(47, 0.39),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.axis()
plt.title('cv/cvm_CP_3')
plt.ylim(0.3,1)
plt.xlim(-1, 56)


####################### combinations of four attributs
ax = plt.subplot(4,1,3)
plt.errorbar(range(len(val1[3])), val1[3], yerr=valm1[3], marker='.', color = 'y')
plt.errorbar(range(len(val1[3])), val2[3], yerr=valm2[3], marker='.', color = 'c')
plt.errorbar(range(len(val1[3])), val3[3], yerr=valm3[3], marker='.')
plt.errorbar(range(len(val1[3])), val4[3], yerr=valm4[3], marker='.', color = 'g')

ax.annotate('K-DurA-domf-E', xy=(5, 0.64), xytext=(5, 0.32), fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Centf-K-DurA-domf', xy=(10, 0.59), xytext=(10, 0.35),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Centf-Dur-DurA-domf', xy=(22, 0.60), xytext=(22, 0.40),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Centf-A-domf-maxAmean', xy=(40, 0.59), xytext=(40, 0.32),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Dur-A-domf-E', xy=(51, 0.65), xytext=(51, 0.36),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Dur-A-domf-E', xy=(61, 0.59), xytext=(61, 0.45),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Dur-A-domf-E', xy=(56, 0.55), xytext=(56, 0.39),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
#plt.legend(['RVL', 'FJS', 'GPS', 'SNE'])
plt.axis()
plt.title('cv/cvm_LB_4')
plt.ylim(0.3,1)
plt.xlim(-1,69)


ax = plt.subplot(4,1,4)
plt.errorbar(range(len(val5[3])), val5[3], yerr=valm5[3], marker='.', color = 'black')
plt.errorbar(range(len(val5[3])), val6[3], yerr=valm6[3], marker='.', color = 'c')
plt.errorbar(range(len(val5[3])), val7[3], yerr=valm7[3], marker='.', color = 'r')
plt.errorbar(range(len(val5[3])), val8[3], yerr=valm8[3], marker='.', color = 'g')

ax.annotate('K-DurA-domf-E', xy=(5, 0.60), xytext=(5, 0.32), fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Centf-K-DurA-domf', xy=(10, 0.60), xytext=(10, 0.35),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Centf-Dur-DurA-domf', xy=(18, 0.60), xytext=(18, 0.40),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Centf-A-domf-maxAmean', xy=(36, 0.59), xytext=(36, 0.40),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Dur-A-domf-E', xy=(51, 0.60), xytext=(51, 0.45),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
            

#plt.legend(['BOR', 'DSO', 'ENO', 'PHR'])
plt.axis()
plt.title('cv/cvm_CP_4')
plt.ylim(0.3,1)
plt.xlim(-1,69)


fig.savefig('cv_cvm_3_4.png')
plt.clf()




######################## combinations of five attributs

fig = plt.figure(figsize = (10,15))
fig.subplots_adjust(left = 0.06, bottom = 0.07,
                     right = 0.8, top = 0.96, wspace = 0.5, hspace = 0.5)

ax = plt.subplot(4,1,1)

plt.errorbar(range(len(val1[4])), val1[4], yerr=valm1[4], marker='.', color = 'y')
plt.errorbar(range(len(val1[4])), val2[4], yerr=valm2[4], marker='.', color = 'c')
plt.errorbar(range(len(val1[4])), val3[4], yerr=valm3[4], marker='.')
plt.errorbar(range(len(val1[4])), val4[4], yerr=valm4[4], marker='.', color = 'g')


# Put a legend to the right of the current axis
ax.legend(['RVL', 'FJS', 'GPS', 'SNE'], loc='center left', bbox_to_anchor=(1, 0.5))

ax.annotate('Centf-Dur-K-DurA-maxAmean', xy=(11, 0.67), xytext=(11, 0.61), fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('K-A-domf-E-maxAmean', xy=(38, 0.62), xytext=(30, 0.53),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('K-DurA-domf-E-maxAmean', xy=(51, 0.65), xytext=(51, 0.50),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Centf-Dur-A-domf-maxAmean', xy=(53, 0.62), xytext=(53, 0.53),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))


plt.axis()
plt.title('cv/cvm_LB_5')
plt.ylim(0.6,1)
plt.xlim(-1,57)


ax = plt.subplot(4,1,2)
plt.errorbar(range(len(val5[4])), val5[4], yerr=valm5[4], marker='.', color = 'black')
plt.errorbar(range(len(val5[4])), val6[4], yerr=valm6[4], marker='.', color = 'c')
plt.errorbar(range(len(val5[4])), val7[4], yerr=valm7[4], marker='.', color = 'r')
plt.errorbar(range(len(val5[4])), val8[4], yerr=valm8[4], marker='.', color = 'g')

ax.annotate('centf-Dur-K-A-E', xy=(0, 0.6), xytext=(0, 0.43), fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Centf-Dur-K-DurA-maxAmean', xy=(11, 0.59), xytext=(11, 0.40), fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('centf-K-DurA-A-E', xy=(16, 0.6), xytext=(16, 0.51),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Centf-Dur-A-E-maxAmean', xy=(41, 0.6), xytext=(41, 0.43),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('K-DurA-domf-E-maxAmean', xy=(51, 0.62), xytext=(51, 0.51),fontsize = 10,
            arrowprops=dict(facecolor='black', shrink=0.05))

# Put a legend to the right of the current axis
ax.legend(['BOR', 'DSO', 'ENO', 'PHR'], loc='center left', bbox_to_anchor=(1, 0.5))

plt.axis()
plt.title('cv/cvm_CP_5')
plt.ylim(0.5,1)
plt.xlim(-1,57)


####################### combinations of six attributs
ax = plt.subplot(4,1,3)
plt.errorbar(range(len(val1[5])), val1[5], yerr=valm1[5], marker='.', color = 'y')
plt.errorbar(range(len(val1[5])), val2[5], yerr=valm2[5], marker='.', color = 'c')
plt.errorbar(range(len(val1[5])), val3[5], yerr=valm3[5], marker='.')
plt.errorbar(range(len(val1[5])), val4[5], yerr=valm4[5], marker='.', color = 'g')

ax.annotate('centf-Dur-DurA-A-domf-E-maxAmean', xy=(4, 0.72), xytext=(4, 0.61),
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Centf-K-DurA-domf-E-maxAmean', xy=(6, 0.77), xytext=(6, 0.64),
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Centf-K-A-domf-E-maxAmean', xy=(14, 0.74), xytext=(14, 0.71),
            arrowprops=dict(facecolor='black', shrink=0.05))

#plt.legend(['RVL', 'FJS', 'GPS', 'SNE'])
plt.axis()
plt.title('cv/cvm_LB_6')
plt.ylim(0.7,1)
plt.xlim(-1,28)

ax = plt.subplot(4,1,4)
plt.errorbar(range(len(val5[5])), val5[5], yerr=valm5[5], marker='.', color = 'black')
plt.errorbar(range(len(val5[5])), val6[5], yerr=valm6[5], marker='.', color = 'c')
plt.errorbar(range(len(val5[5])), val7[5], yerr=valm7[5], marker='.', color = 'r')
plt.errorbar(range(len(val5[5])), val8[5], yerr=valm8[5], marker='.', color = 'g')


ax.annotate('centf-Dur-DurA-A-domf-E-maxAmean', xy=(4, 0.63), xytext=(4, 0.45),
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('K-DurA-A-domf-E-maxAmean', xy=(19, 0.69), xytext=(19, 0.61),
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Centf-Dur-K-DurA-domf-maxAmean', xy=(11, 0.66), xytext=(11, 0.54),
            arrowprops=dict(facecolor='black', shrink=0.05))

#plt.legend(['BOR', 'DSO', 'ENO', 'PHR'])
plt.axis()
plt.title('cv/cvm_CP_6')
plt.ylim(0.6,1)
plt.xlim(-1,28)

fig.savefig('cv_cvm_5_6.png')
plt.clf()


####################### combinations of six attributs
ax = plt.subplot(4,1,1)
plt.errorbar(range(len(val1[6])), val1[6], yerr=valm1[6], marker='.', color = 'y')
plt.errorbar(range(len(val1[6])), val2[6], yerr=valm2[6], marker='.', color = 'c')
plt.errorbar(range(len(val1[6])), val3[6], yerr=valm3[6], marker='.')
plt.errorbar(range(len(val1[6])), val4[6], yerr=valm4[6], marker='.', color = 'g')

# Put a legend to the right of the current axis
ax.legend(['RVL', 'FJS', 'GPS', 'SNE'], loc='center left', bbox_to_anchor=(1, 0.5))

plt.axis()
plt.title('cv/cvm_LB_7')
plt.ylim(0.7,1)
plt.xlim(-1, 8)

ax = plt.subplot(4,1,2)
plt.errorbar(range(len(val5[6])), val5[6], yerr=valm5[6], marker='.', color = 'black')
plt.errorbar(range(len(val5[6])), val6[6], yerr=valm6[6], marker='.', color = 'c')
plt.errorbar(range(len(val5[6])), val7[6], yerr=valm7[6], marker='.', color = 'r')
plt.errorbar(range(len(val5[6])), val8[6], yerr=valm8[6], marker='.', color = 'g')

# Put a legend to the right of the current axis
ax.legend(['BOR', 'DSO', 'ENO', 'PHR'], loc='center left', bbox_to_anchor=(1, 0.5))
ax.text(1, 0.65, 'centf-Dur-DurA-A-domf-E-maxAmean', style='italic',
        bbox={'facecolor':'red', 'alpha':0, 'pad':10})

#plt.legend(['BOR', 'DSO', 'ENO', 'PHR'])
plt.axis()
plt.title('cv/cvm_CP_7')
plt.ylim(0.6,1)
plt.xlim(-1, 8)


fig.savefig('cv_cvm_7.png')
plt.clf()


ax = plt.subplot(111)



#plt.legend(['BOR', 'DSO', 'ENO', 'PHR'])
plt.axis()

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

# Put a legend to the right of the current axis
ax.legend(['BOR', 'DSO', 'ENO', 'PHR'], loc='center left', bbox_to_anchor=(1, 1))




###################### LB + CP

fig = plt.figure(figsize = (20, 10))
plt.errorbar(range(len(val1[1])), val1[1], yerr=valm1[1], marker='.', color = 'y')
plt.errorbar(range(len(val1[1])), val2[1], yerr=valm2[1], marker='.', color = 'c')
plt.errorbar(range(len(val1[1])), val3[1], yerr=valm3[1], marker='.')
plt.errorbar(range(len(val1[1])), val4[1], yerr=valm4[1], marker='.', color = 'g')

plt.errorbar(range(len(val5[1])), val5[1], yerr=valm5[1], marker='.', color = 'black')
plt.errorbar(range(len(val5[1])), val6[1], yerr=valm6[1], marker='.', color = 'orange')
plt.errorbar(range(len(val5[1])), val7[1], yerr=valm7[1], marker='.', color = 'r')
plt.errorbar(range(len(val5[1])), val8[1], yerr=valm8[1], marker='.', color = 'grey')


plt.legend(['BOR', 'DSO', 'ENO', 'PHR', 'RVL', 'FJS', 'GPS', 'SNE'])
plt.axis()
plt.title('cv/cvm_LB_CP')
plt.ylim(0.3,1)

fig.savefig('cv_cvm_LB_CP.png')
plt.clf()



# give the axe legend 

# for the plot with the combination of one attribut
I = ['[8]', '[6]', '[7]', '[4]', '[5]', '[2]', '[0]', '[1]']
#etc ...
II = ['[4, 2]', '[6, 2]', '[5, 0]', '[0, 8]', '[6, 0]', '[7, 5]', '[4, 8]', '[1, 5]', '[5, 8]', '[1, 7]', '[6, 5]', '[5, 2]', '[6, 8]', '[4, 0]', '[4, 6]', '[7, 0]', '[2, 8]', '[1, 8]', '[7, 2]', '[7, 6]', '[7, 8]', '[1, 2]', '[0, 2]', '[1, 0]', '[4, 5]', '[1, 4]', '[1, 6]', '[4, 7]']
III = ['[1, 4, 5]', '[1, 6, 8]', '[7, 5, 0]', '[1, 4, 7]', '[1, 2, 8]', '[1, 6, 0]', '[4, 7, 5]', '[1, 6, 2]', '[7, 0, 2]', '[6, 0, 8]', '[4, 7, 2]', '[4, 7, 0]', '[5, 0, 2]', '[1, 5, 8]', '[7, 5, 8]', '[4, 5, 0]', '[5, 0, 8]', '[7, 6, 0]', '[4, 0, 2]', '[1, 7, 0]', '[6, 5, 8]', '[1, 7, 6]', '[7, 5, 2]', '[6, 5, 2]', '[7, 2, 8]', '[4, 7, 8]', '[4, 6, 5]', '[4, 5, 2]', '[6, 5, 0]', '[7, 6, 8]', '[1, 6, 5]', '[1, 4, 6]', '[1, 7, 8]', '[4, 6, 2]', '[1, 4, 0]', '[1, 0, 2]', '[4, 6, 0]', '[0, 2, 8]', '[1, 7, 2]', '[1, 5, 2]', '[1, 0, 8]', '[4, 7, 6]', '[7, 6, 2]', '[1, 5, 0]', '[6, 2, 8]', '[4, 5, 8]', '[1, 4, 8]', '[1, 7, 5]', '[4, 6, 8]', '[7, 6, 5]', '[7, 0, 8]', '[5, 2, 8]', '[4, 2, 8]', '[4, 0, 8]', '[1, 4, 2]', '[6, 0, 2]']
IV = ['[1, 5, 2, 8]', '[7, 6, 2, 8]', '[4, 7, 0, 8]', '[1, 7, 0, 8]', '[1, 4, 7, 5]', '[7, 6, 0, 2]', '[4, 7, 6, 2]', '[4, 7, 5, 2]', '[4, 7, 0, 2]', '[1, 7, 5, 0]', '[1, 7, 6, 0]', '[4, 5, 2, 8]', '[4, 7, 5, 0]', '[1, 4, 6, 2]', '[7, 5, 0, 2]', '[1, 7, 5, 2]', '[1, 7, 6, 2]', '[4, 6, 0, 8]', '[1, 4, 6, 0]', '[1, 6, 5, 0]', '[1, 4, 2, 8]', '[1, 4, 5, 8]', '[1, 6, 5, 2]', '[1, 4, 0, 8]', '[4, 7, 6, 8]', '[4, 7, 2, 8]', '[4, 7, 5, 8]', '[7, 6, 5, 8]', '[7, 6, 0, 8]', '[4, 7, 6, 0]', '[4, 6, 5, 0]', '[7, 6, 5, 2]', '[1, 4, 0, 2]', '[1, 7, 0, 2]', '[1, 0, 2, 8]', '[4, 6, 0, 2]', '[6, 5, 0, 2]', '[1, 5, 0, 8]', '[7, 6, 5, 0]', '[5, 0, 2, 8]', '[4, 7, 6, 5]', '[7, 5, 2, 8]', '[1, 6, 0, 8]', '[1, 4, 6, 5]', '[6, 5, 2, 8]', '[6, 5, 0, 8]', '[6, 0, 2, 8]', '[1, 6, 5, 8]', '[4, 6, 2, 8]', '[4, 0, 2, 8]', '[1, 4, 7, 8]', '[1, 5, 0, 2]', '[4, 5, 0, 2]', '[4, 6, 5, 8]', '[1, 4, 5, 0]', '[1, 4, 6, 8]', '[7, 5, 0, 8]', '[1, 6, 0, 2]', '[1, 7, 2, 8]', '[1, 6, 2, 8]', '[1, 4, 7, 6]', '[7, 0, 2, 8]', '[4, 5, 0, 8]', '[1, 4, 5, 2]', '[1, 4, 7, 0]', '[1, 7, 6, 5]', '[1, 4, 7, 2]', '[1, 7, 5, 8]', '[1, 7, 6, 8]', '[4, 6, 5, 2]']
V = ['[1, 4, 7, 5, 2]', '[1, 4, 7, 0, 2]', '[1, 7, 5, 0, 8]', '[1, 5, 0, 2, 8]', '[1, 4, 6, 2, 8]', '[1, 7, 6, 2, 8]', '[4, 6, 5, 2, 8]', '[4, 7, 5, 0, 8]', '[4, 7, 6, 0, 2]', '[1, 4, 6, 5, 2]', '[1, 4, 0, 2, 8]', '[1, 4, 7, 6, 8]', '[1, 6, 0, 2, 8]', '[4, 7, 5, 2, 8]', '[1, 6, 5, 0, 2]', '[1, 4, 7, 5, 0]', '[1, 7, 6, 5, 2]', '[1, 7, 6, 0, 2]', '[1, 4, 7, 0, 8]', '[1, 7, 0, 2, 8]', '[7, 6, 5, 0, 2]', '[1, 4, 6, 0, 8]', '[1, 4, 7, 6, 5]', '[4, 7, 6, 5, 8]', '[1, 4, 6, 0, 2]', '[4, 7, 6, 5, 0]', '[1, 4, 7, 5, 8]', '[7, 6, 5, 2, 8]', '[1, 7, 5, 0, 2]', '[1, 7, 6, 5, 0]', '[1, 6, 5, 0, 8]', '[4, 7, 0, 2, 8]', '[1, 7, 6, 0, 8]', '[4, 6, 0, 2, 8]', '[6, 5, 0, 2, 8]', '[1, 4, 5, 0, 2]', '[4, 7, 5, 0, 2]', '[4, 6, 5, 0, 8]', '[7, 5, 0, 2, 8]', '[1, 7, 6, 5, 8]', '[7, 6, 5, 0, 8]', '[1, 4, 5, 2, 8]', '[4, 6, 5, 0, 2]', '[4, 7, 6, 0, 8]', '[1, 4, 7, 2, 8]', '[1, 4, 7, 6, 2]', '[1, 4, 6, 5, 0]', '[4, 7, 6, 2, 8]', '[1, 6, 5, 2, 8]', '[1, 4, 6, 5, 8]', '[4, 5, 0, 2, 8]', '[7, 6, 0, 2, 8]', '[1, 7, 5, 2, 8]', '[1, 4, 5, 0, 8]', '[4, 7, 6, 5, 2]', '[1, 4, 7, 6, 0]']
VI = ['[1, 4, 5, 0, 2, 8]', '[1, 4, 7, 0, 2, 8]', '[1, 7, 6, 5, 0, 8]', '[1, 7, 6, 5, 2, 8]', '[1, 4, 6, 0, 2, 8]', '[4, 7, 6, 0, 2, 8]', '[1, 7, 6, 0, 2, 8]', '[4, 6, 5, 0, 2, 8]', '[1, 4, 6, 5, 2, 8]', '[1, 4, 7, 6, 5, 2]', '[1, 4, 7, 6, 5, 0]', '[4, 7, 6, 5, 2, 8]', '[1, 4, 7, 6, 0, 8]', '[1, 4, 6, 5, 0, 8]', '[1, 7, 5, 0, 2, 8]', '[1, 4, 7, 6, 0, 2]', '[1, 4, 6, 5, 0, 2]', '[1, 4, 7, 5, 0, 2]', '[4, 7, 6, 5, 0, 2]', '[7, 6, 5, 0, 2, 8]', '[1, 4, 7, 6, 2, 8]', '[1, 6, 5, 0, 2, 8]', '[1, 4, 7, 5, 0, 8]', '[4, 7, 5, 0, 2, 8]', '[1, 4, 7, 5, 2, 8]', '[1, 4, 7, 6, 5, 8]', '[1, 7, 6, 5, 0, 2]', '[4, 7, 6, 5, 0, 8]']
VII = ['[4, 7, 6, 5, 0, 2, 8]', '[1, 4, 7, 6, 5, 0, 8]', '[1, 4, 6, 5, 0, 2, 8]', '[1, 4, 7, 6, 0, 2, 8]', '[1, 4, 7, 6, 5, 2, 8]', '[1, 7, 6, 5, 0, 2, 8]', '[1, 4, 7, 6, 5, 0, 2]', '[1, 4, 7, 5, 0, 2, 8]']
VIII = ['[1, 4, 7, 6, 5, 0, 2, 8]']

att = ['dom_f', 'cent_f', 'E', 'AsDec', 'DUR', 'A', 'dur/A', 'K', 'maxA_mean', 'dom_f', 'cent_f', 'E', 'AsDec', 'DUR', 'A', 'dur/A', 'K', 'maxA_mean']
# for 'O', it's 'dom_f'
# etc...

