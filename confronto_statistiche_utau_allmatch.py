import numpy as np
from matplotlib import pyplot as plt,cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 

reference = np.genfromtxt('chann1000.dat')
y_plus = reference[:,1]
u_plus = reference[:,2]

tauwall_dns = dict(np.load('dataset_dict/tauwall_scaled_test.npz'))
tauwall_60 = dict(np.load('tauwall_EWM60_test.npz'))

u60 = dict(np.load('slice_u60_less_test.npz'))

utau_60 = {}

for key60 in utau_60.keys():
    utau_60[key60] = np.sqrt(utau_60[key60])
    

tauwall_dns_flat    = []
tauwall_60_flat     = []
tauwall_60_EWM_flat = []

for (key, key60) in zip(tauwall_dns.keys(), tauwall_10.keys(), tauwall_30.keys(), tauwall_60.keys(), tauwall_120_EWM.keys()):
    campodns = tauwall_dns[key].ravel()
    campo10 = tauwall_10[key10].ravel()
    campo30 = tauwall_30[key30].ravel()
    campo60 = tauwall_60[key60].ravel()
    campo10_EWM = tauwall_10_EWM[key10].ravel()
    campo30_EWM = tauwall_30_EWM[key30].ravel()
    campo60_EWM = tauwall_60_EWM[key60].ravel()
    campo120_EWM = tauwall_120_EWM[key120].ravel()
    tauwall_dns_flat.append(campodns)
    tauwall_10_flat.append(campo10)
    tauwall_30_flat.append(campo30)
    tauwall_60_flat.append(campo60)
    tauwall_10_EWM_flat.append(campo10_EWM)
    tauwall_30_EWM_flat.append(campo30_EWM)
    tauwall_60_EWM_flat.append(campo60_EWM)
    tauwall_120_EWM_flat.append(campo120_EWM)

tauwall_dns_flat    = np.array(tauwall_dns_flat)
tauwall_10_flat     = np.array(tauwall_10_flat)
tauwall_30_flat     = np.array(tauwall_30_flat)
tauwall_60_flat     = np.array(tauwall_60_flat)
tauwall_10_EWM_flat = np.array(tauwall_10_EWM_flat)
tauwall_30_EWM_flat = np.array(tauwall_30_EWM_flat)
tauwall_60_EWM_flat = np.array(tauwall_60_EWM_flat)    
tauwall_120_EWM_flat = np.array(tauwall_120_EWM_flat)    
    
tauwall_dns_mean    = np.mean(tauwall_dns_flat)
tauwall_10_mean     = np.mean(tauwall_10_flat)
tauwall_30_mean     = np.mean(tauwall_30_flat)
tauwall_60_mean     = np.mean(tauwall_60_flat)
tauwall_10_EWM_mean = np.mean(tauwall_10_EWM_flat)
tauwall_30_EWM_mean = np.mean(tauwall_30_EWM_flat)
tauwall_60_EWM_mean = np.mean(tauwall_60_EWM_flat)
tauwall_120_EWM_mean = np.mean(tauwall_120_EWM_flat)

utau_dns_mean    = np.sqrt(tauwall_dns_mean)
utau_10_mean     = np.sqrt(tauwall_10_mean)
utau_30_mean     = np.sqrt(tauwall_30_mean)
utau_60_mean     = np.sqrt(tauwall_60_mean)
utau_10_EWM_mean = np.sqrt(tauwall_10_EWM_mean)
utau_30_EWM_mean = np.sqrt(tauwall_30_EWM_mean)
utau_60_EWM_mean = np.sqrt(tauwall_60_EWM_mean)
utau_120_EWM_mean = np.sqrt(tauwall_120_EWM_mean)

#print(utau_dns_mean)
#print(utau_10_mean)
#print(utau_30_mean)
#print(utau_60_mean)
#print(utau_10_EWM_mean)
#print(utau_30_EWM_mean)
#print(utau_60_EWM_mean)

u10_flat = []
u30_flat = []
u60_flat = []
u120_flat = []
for (key10, key30, key60, key120) in zip(tauwall_10.keys(), tauwall_30.keys(), tauwall_60.keys(), u120.keys()):
    campo10 = u10[key10].ravel()
    campo30 = u30[key30].ravel()
    campo60 = u60[key60].ravel()
    campo120 = u120[key120].ravel()
    u10_flat.append(campo10)
    u30_flat.append(campo30)
    u60_flat.append(campo60)
    u120_flat.append(campo120)

u10_flat = np.array(u10_flat)
u30_flat = np.array(u30_flat)
u60_flat = np.array(u60_flat)
u120_flat = np.array(u120_flat)

u10_mean = np.mean(u10_flat)
u30_mean = np.mean(u30_flat)
u60_mean = np.mean(u60_flat)
u120_mean = np.mean(u120_flat)

#print(u10_mean)
#print(u30_mean)
#print(u60_meannp.arra([])
u10_plusNN  = u10_mean / utau_10_mean
u30_plusNN  = u30_mean / utau_30_mean
u60_plusNN  = u60_mean / utau_60_mean
u10_plusEWM = u10_mean / utau_10_EWM_mean
u30_plusEWM = u30_mean / utau_30_EWM_mean
u60_plusEWM = u60_mean / utau_60_EWM_mean
u120_plusEWM = u120_mean / utau_120_EWM_mean
udns10_plus = u10_mean / utau_dns_mean
udns30_plus = u30_mean / utau_dns_mean
udns60_plus = u60_mean / utau_dns_mean
    
#u10_plusNN  = np.array(u10_plusNN)
#u30_plusNN  = np.array(u30_plusNN)
#u60_plusNN  = np.array(u60_plusNN)
#u10_plusEWM = np.array(u10_plusEWM)
#u30_plusEWM = np.array(u30_plusEWM)
#u60_plusEWM = np.array(u60_plusEWM)
#udns10_plus = np.array(udns10_plus)
#udns30_plus = np.array(udns30_plus)
#udns60_plus = np.array(udns60_plus)
#
#u10_plusNN  = np.mean(u10_plusNN)
#u30_plusNN  = np.mean(u30_plusNN)
#u60_plusNN  = np.mean(u60_plusNN)
#u10_plusEWM = np.mean(u10_plusEWM)
#u30_plusEWM = np.mean(u30_plusEWM)
#u60_plusEWM = np.mean(u60_plusEWM)
#udns10_plus = np.mean(udns10_plus)
#udns30_plus = np.mean(udns30_plus)
#udns60_plus = np.mean(udns60_plus)

y60_plus = 60.08545460723399
y120_plus = 119.02922741466598
y30_plus = 30.386406921228
y10_plus = 10.193845712224201

fig,ax = plt.subplots()
ax.semilogx(y_plus,u_plus, color='black',label='reference')
ax.scatter(y10_plus, u10_plusNN, label='NN utau10')
ax.scatter(y30_plus, u30_plusNN, label= 'NN utau30')
ax.scatter(y60_plus, u60_plusNN, label='NN utau60')
ax.scatter(y10_plus, u10_plusEWM, label= 'EWM utau10')
ax.scatter(y30_plus, u30_plusEWM, label= 'EWM utau30')
ax.scatter(y60_plus, u60_plusEWM, label= 'EWM utau60')
ax.scatter(y120_plus, u120_plusEWM, label= 'EWM utau120')
ax.scatter(y10_plus, udns10_plus, label= 'DNS utau10')
ax.scatter(y30_plus, udns30_plus, label= 'DNS utau30')
ax.scatter(y60_plus, udns60_plus, label= 'DNS utau60')
#for i in range(mean_u60_slice.shape[0]):
#    ax.scatter(zc_cord_plus[id1], mean_u60_slice[i] / utau_60)
ax.set_xlabel('$y^+$')
ax.set_ylabel('$U/utau$')
ax.set_title('$U^+ vs, y^+$')
plt.grid()
plt.legend()
plt.show()
plt.close()

uflut_10 = np.array([])
uflut_30 = np.array([])
uflut_60 = np.array([])
uflut_120 = np.array([])
u10_square = np.array([])
u30_square = np.array([])
u60_square = np.array([])
u120_square = np.array([])

for (key10, key30, key60,key120) in zip(u10.keys(), u30.keys(), u60.keys(), u120.keys()):
    u10_square = np.append(u10_square,((u10[key10].ravel())**2))
    u30_square = np.append(u30_square,((u30[key30].ravel())**2))
    u60_square = np.append(u60_square,((u60[key60].ravel())**2))
    u120_square = np.append(u120_square,((u120[key120].ravel())**2))
    campo10 = u10[key10].ravel()
    flut10 = campo10 - u10_mean
    campo30 = u30[key30].ravel()
    flut30 = campo30 - u30_mean
    campo60 = u60[key60].ravel()
    flut60 = campo60 - u60_mean
    campo120 = u120[key120].ravel()
    flut120 = campo120 - u120_mean
    uflut_10 = np.append(uflut_10, flut10)
    uflut_30 = np.append(uflut_30, flut30)
    uflut_60 = np.append(uflut_60, flut60)
    uflut_120 = np.append(uflut_120, flut120)
    

u10_square_mean = np.mean(u10_square)
u30_square_mean = np.mean(u30_square)
u60_square_mean = np.mean(u60_square)
u120_square_mean = np.mean(u120_square)



uflut_10_square = np.square(uflut_10)
uflut_30_square = np.square(uflut_30)
uflut_60_square = np.square(uflut_60)
uflut_120_square = np.square(uflut_120)

uflut_10_square_mean = np.mean(uflut_10_square)
uflut_30_square_mean = np.mean(uflut_30_square)
uflut_60_square_mean = np.mean(uflut_60_square)
uflut_120_square_mean = np.mean(uflut_120_square)

utau_dns_mean_square    = utau_dns_mean**2
utau_10_mean_square     = utau_10_mean**2
utau_30_mean_square     = utau_30_mean**2
utau_60_mean_square     = utau_60_mean**2
utau_10_EWM_mean_square = utau_10_EWM_mean**2
utau_30_EWM_mean_square = utau_30_EWM_mean**2
utau_60_EWM_mean_square = utau_60_EWM_mean**2
utau_120_EWM_mean_square = utau_120_EWM_mean**2

uplus_rms10_dns = uflut_10_square_mean / utau_dns_mean_square
uplus_rms30_dns = uflut_30_square_mean / utau_dns_mean_square
uplus_rms60_dns = uflut_60_square_mean / utau_dns_mean_square
uplus_rms10_NN = uflut_10_square_mean / utau_10_mean_square
uplus_rms30_NN = uflut_30_square_mean / utau_30_mean_square
uplus_rms60_NN = uflut_60_square_mean / utau_60_mean_square
uplus_rms10_EWM = uflut_10_square_mean / utau_10_EWM_mean_square
uplus_rms30_EWM = uflut_30_square_mean / utau_30_EWM_mean_square
uplus_rms60_EWM = uflut_60_square_mean / utau_60_EWM_mean_square 
uplus_rms120_EWM = uflut_120_square_mean / utau_120_EWM_mean_square 

#uplus_rms10_dns = u10_square_mean - utau_dns_mean_square
#uplus_rms30_dns = u30_square_mean - utau_dns_mean_square
#uplus_rms60_dns = u60_square_mean - utau_dns_mean_square
#uplus_rms10_NN  = u10_square_mean - utau_10_mean_square
#uplus_rms30_NN  = u30_square_mean - utau_30_mean_square
#uplus_rms60_NN  = u60_square_mean - utau_60_mean_square
#uplus_rms10_EWM = u10_square_mean - utau_10_EWM_mean_square
#uplus_rms30_EWM = u30_square_mean - utau_30_EWM_mean_square
#uplus_rms60_EWM = u60_square_mean - utau_60_EWM_mean_square 

#uu10_plusNN = []
#uu30_plusNN = []
#uu60_plusNN = []
#uu10_plusEWM = []
#uu30_plusEWM = []
#uu60_plusEWM = []
#uudns10_plus = []
#uudns30_plus = []
#uudns60_plus = []
#for (key, key10, key30, key60) in zip(utau_dns.keys(), utau_10.keys(), utau_30.keys(), utau_60.keys()):
#    uplusdns10 = (np.var(u10[key10].ravel())) /( np.square(np.mean(utau_dns[key].ravel())))
#    uplusdns30 = (np.var(u30[key30].ravel())) /( np.square(np.mean(utau_dns[key].ravel())))
#    uplusdns60 = (np.var(u60[key60].ravel())) /( np.square(np.mean(utau_dns[key].ravel())))
#    uplusEWM10 = (np.var(u10[key10].ravel())) /( np.square(np.mean(utau_10_EWM[key10].ravel())))
#    uplusEWM30 = (np.var(u30[key30].ravel())) /( np.square(np.mean(utau_30_EWM[key30].ravel())))
#    uplusEWM60 = (np.var(u60[key60].ravel())) /( np.square(np.mean(utau_60_EWM[key60].ravel())))
#    uplusNN10 = (np.var(u10[key10].ravel())) / (np.square(np.mean(utau_10[key10].ravel())))
#    uplusNN30 = (np.var(u30[key30].ravel())) / (np.square(np.mean(utau_30[key30].ravel())))
#    uplusNN60 = (np.var(u60[key60].ravel())) / (np.square(np.mean(utau_60[key60].ravel())))
#    uudns10_plus.append(uplusdns10)
#    uudns30_plus.append(uplusdns30)
#    uudns60_plus.append(uplusdns60)
#    uu10_plusEWM.append(uplusEWM10)
#    uu30_plusEWM.append(uplusEWM30)
#    uu60_plusEWM.append(uplusEWM60)
#    uu10_plusNN.append(uplusNN10)
#    uu30_plusNN.append(uplusNN30)
#    uu60_plusNN.append(uplusNN60)
#
#    
#uu10_plusNN  = np.array(u10_plusNN)
#uu30_plusNN  = np.array(u30_plusNN)
#uu60_plusNN  = np.array(u60_plusNN)
#uu10_plusEWM = np.array(u10_plusEWM)
#uu30_plusEWM = np.array(u30_plusEWM)
#uu60_plusEWM = np.array(u60_plusEWM)
#uudns10_plus = np.array(udns10_plus)
#uudns30_plus = np.array(udns30_plus)
#uudns60_plus = np.array(udns60_plus)
#
#uu10_plusNN  = np.mean(u10_plusNN)
#uu30_plusNN  = np.mean(u30_plusNN)
#uu60_plusNN  = np.mean(u60_plusNN)
#uu10_plusEWM = np.mean(u10_plusEWM)
#uu30_plusEWM = np.mean(u30_plusEWM)
#uu60_plusEWM = np.mean(u60_plusEWM)
#uudns10_plus = np.mean(udns10_plus)
#uudns30_plus = np.mean(udns30_plus)
#uudns60_plus = np.mean(udns60_plus)

uu = reference[:,3]
vv = reference[:,4]
ww = reference[:,5]
fig,ax = plt.subplots()
ax.semilogx(y_plus, uu**2, color='black', label='var w')
ax.scatter(y10_plus, uplus_rms10_NN , label='NN utau10')
ax.scatter(y30_plus, uplus_rms30_NN , label= 'NN utau30')
ax.scatter(y60_plus, uplus_rms60_NN , label='NN utau60')
ax.scatter(y10_plus, uplus_rms10_EWM ,  label= 'EWM utau10')
ax.scatter(y30_plus, uplus_rms30_EWM ,  label= 'EWM utau30')
ax.scatter(y60_plus, uplus_rms60_EWM ,  label= 'EWM utau60')
ax.scatter(y120_plus, uplus_rms120_EWM ,  label= 'EWM utau120')
ax.scatter(y10_plus, uplus_rms10_dns ,  label= 'DNS utau10')
ax.scatter(y30_plus, uplus_rms30_dns ,  label= 'DNS utau30')
ax.scatter(y60_plus, uplus_rms60_dns ,  label= 'DNS utau60')
ax.set_xlabel('$y^+$')
ax.set_xlim(left=1)
ax.set_ylabel('$var/u_{tau}$')
plt.legend()
plt.show()
plt.close()

