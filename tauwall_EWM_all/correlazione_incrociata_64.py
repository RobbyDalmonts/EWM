import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dns = np.load('tauwall_target_64.npy')
ewm = np.load('tauwall_EWM120_64.npy')
u60 = np.load('slice_u120_64.npy')
#model = np.load('tauwall_ML_model.npz')
model1 = np.load('tauwall_ML_model_t820_r102_120.npz')
#model2 = np.load('tauwall_ML_model_t300_r180.npz')
#model3 = np.load('tauwall_ML_model_t240_r30.npz')
#model4 = np.load('tauwall_ML_model_customized.npy')
#uncond = np.load('model_reconstruction.npy')

#model_unc = {}
#for i in range(uncond.shape[0]):
#    key = '{}'.format(i)
#    model_unc[key] = uncond[i][0]
#
#
#alpha = 0.1388384832398331
#alpha_max = 0.25370291
#alpha_max = 0.5
#linear = {}
#for (i,k) in zip(range(uncond.shape[0]), ewm.keys()):
#    campo_e = ewm[k]
#    model_u = uncond[i][0]
#    campo = alpha * campo_e + (1 - alpha) * model_u
#    linear[k] = campo
#
#
nx = 1216
nz = 640
delta = 0.5
lx = 5
lz = 1.5
dx = lx / (nx-1)
dz = lz / (nz-1)
#
#len_dns = len(dns.keys())
#len_ewm = len(ewm.keys())
#len_u60 = len(u60.keys())
#len_model = len(model.keys())
len_model1 = len(model1.keys())
#len_model2 = len(model2.keys())
#len_model3 = len(model3.keys())
#len_model4 = len(model4.keys())
#len_uncond = len(model_unc.keys())
#
#
##in riferimento ai dati normalizzati con standard scaler sui valori del train dataset
#noise = np.random.randn(64,64)
#attenuation_300 = 0.6296187
#noise_level_300 = 0.7769043
#attenuation_240 = 0.74220026
#noise_level_240 = 0.6701782
#noise_level_600 = 0.9869755
#attenuation_600 = 0.1608707
#
#data_mean= 0.002453068295866685 #statistiche del training set 
#data_std= 0.0010229427221970142
#
#
#dns_all = []
#ewm_all = []
#ewm_all_t300 = []
#ewm_all_t240 = []
#ewm_all_t600 = []
#u60_all = []
#model_all = []
#model1_all = []
#model2_all = []
#model3_all = []
#
#ewm_t300 = {}
#ewm_t240 = {}
#ewm_t600 = {}
#for k in ewm.keys():
#    campo = (ewm[k] - data_mean) / data_std
#    campo_300 = campo * attenuation_300 + noise_level_300 * noise
#    campo_300 = campo_300 * data_std + data_mean
#    ewm_t300[k] = campo_300
#    campo_240 = campo * attenuation_240 + noise_level_240 * noise
#    campo_240 = campo_240 * data_std + data_mean
#    ewm_t240[k] = campo_240
#    campo_600 = campo * attenuation_600 + noise_level_600 * noise
#    campo_600 = campo_600 * data_std + data_mean
#    ewm_t600[k] = campo_600
#
#for (kd,ke,ku) in zip(dns.keys(), ewm.keys(), u60.keys()):
#    campo_d = dns[kd].ravel()
#    campo_e = ewm[ke].ravel()
#    campo_e_t300 = ewm_t300[ke].ravel()
#    campo_e_t240 = ewm_t240[ke].ravel()
#    campo_e_t600 = ewm_t600[ke].ravel()
#    campo_u = u60[ku].ravel()
#    dns_all.append(campo_d)
#    ewm_all.append(campo_e)
#    ewm_all_t300.append(campo_e_t300)
#    ewm_all_t240.append(campo_e_t240)
#    ewm_all_t600.append(campo_e_t600)
#    u60_all.append(campo_u)
#
#for k in model.keys():
#    campo = model[k].ravel()
#    model_all.append(campo)
#for k in model1.keys():
#    campo = model1[k].ravel()
#    model1_all.append(campo)
#for k in model2.keys():
#    campo = model2[k].ravel()
#    model2_all.append(campo)
#for k in model3.keys():
#    campo = model3[k].ravel()
#    model3_all.append(campo)
#
#dns_all = np.array(dns_all)
#ewm_all = np.array(ewm_all)
#ewm_all_t300 = np.array(ewm_all_t300)
#ewm_all_t240 = np.array(ewm_all_t240)
#ewm_all_t600 = np.array(ewm_all_t600)
#u60_all = np.array(u60_all)
#model_all = np.array(model_all)
#model1_all = np.array(model1_all)
#model2_all = np.array(model2_all)
#model3_all = np.array(model3_all)
#
#mean_dns = np.mean(dns_all)
#mean_ewm = np.mean(ewm_all)
#mean_ewm_t300 = np.mean(ewm_all_t300)
#mean_ewm_t240 = np.mean(ewm_all_t240)
#mean_ewm_t600 = np.mean(ewm_all_t600)
#mean_u60 = np.mean(u60_all)
#mean_model = np.mean(model_all)
#mean_model1 = np.mean(model1_all)
#mean_model2 = np.mean(model2_all)
#mean_model3 = np.mean(model3_all)
#
#utau_mean_dns = np.sqrt(mean_dns)
#utau_mean_ewm = np.sqrt(mean_ewm)
#utau_mean_ewm_t300 = np.sqrt(mean_ewm_t300)
#utau_mean_ewm_t240 = np.sqrt(mean_ewm_t240)
#utau_mean_ewm_t600 = np.sqrt(mean_ewm_t600)
#utau_mean_model = np.sqrt(mean_model)
#utau_mean_model1 = np.sqrt(mean_model1)
#utau_mean_model2 = np.sqrt(mean_model2)
#utau_mean_model3 = np.sqrt(mean_model3)
#
#
#flut_dns = (dns_all - mean_dns) / mean_dns
#flut_ewm = (ewm_all - mean_ewm) / mean_ewm
#flut_ewm_t300 = (ewm_all_t300 - mean_ewm_t300) / mean_ewm_t300
#flut_ewm_t240 = (ewm_all_t240 - mean_ewm_t240) / mean_ewm_t240
#flut_ewm_t600 = (ewm_all_t600 - mean_ewm_t600) / mean_ewm_t600
#flut_model = (model_all - mean_model) / mean_model
#flut_model1 = (model1_all - mean_model1) / mean_model1
#flut_model2 = (model2_all - mean_model2) / mean_model2
#flut_model3 = (model3_all - mean_model3) / mean_model3
#flut_u60 = (u60_all - mean_u60) / utau_mean_dns
#
#
#coefficiente k che stabilisce la grandezza della finestra per la corss correlation

kmax = 1.5 * delta // dx
delta_x_max = kmax * dx
if delta_x_max > lx / 2:
    kmax = nx // 2

print(kmax)
kmax=64
k_array = np.linspace(-kmax,kmax,2*64+1)

x_dimension = []
for k in k_array:
    a = k* dx / delta
    x_dimension.append(a)
x_dimension = np.array(x_dimension)
print(x_dimension)
print(x_dimension.shape)
len_dns = int(dns.shape[0]*dns.shape[1])
len_ewm = int(dns.shape[0]*dns.shape[1])
Rk_array = np.zeros((len_dns,k_array.shape[0]))
Rk_ewm = np.zeros((len_ewm,k_array.shape[0]))
#Rk_ewm_t300 = np.zeros((len_ewm,k_array.shape[0]))
#Rk_ewm_t240 = np.zeros((len_ewm,k_array.shape[0]))
#Rk_ewm_t600 = np.zeros((len_ewm,k_array.shape[0]))
Rk_model = np.zeros((len_model,k_array.shape[0]))
#Rk_model1 = np.zeros((len_model1,k_array.shape[0]))
#Rk_model2 = np.zeros((len_model2,k_array.shape[0]))
#Rk_model3 = np.zeros((len_model3,k_array.shape[0]))
#Rk_model4 = np.zeros((len_model4,k_array.shape[0]))
#Rk_uncond = np.zeros((len_uncond,k_array.shape[0]))
#Rk_linear = np.zeros((len_uncond,k_array.shape[0]))


nx = 64
nz = 64
from numba import njit,prange

@njit(parallel=True, fastmath=True)
def compute_Rk(fluct_tau, fluct_vel, k_array, nx,nz):
    Nk = k_array.size
    N = nx * nz
    sigma_tau = np.sqrt(np.mean(fluct_tau**2))
    sigma_vel = np.sqrt(np.mean(fluct_vel**2))
    denom = sigma_tau * sigma_vel
    out = np.empty(Nk, dtype=np.float64)

    for kk in prange(Nk):
        k = int(k_array[kk])
        prod = 0.0
        f = 0
        for m in range(nz):
            for i in range(nx):
                j = (i-k) 
                if j > 0 and j < nx :
                    prod = prod + fluct_tau[i,m] * fluct_vel[j,m]
                    f = f + 1
        Ck = prod / f
        out[kk] = Ck / denom if denom > 0.0 else 0.0
    return out
                
f = 0
for i in range(dns.shape[0]):
    for j in range(dns.shape[1]):
        campo = dns[i][j]
        vel = u60[i][j]
        equi = ewm[i][j]
        #equi_300 = ewm_t300[ke]
        #equi_240 = ewm_t240[ke]
        #equi_600 = ewm_t600[ke]

        flut_tau = campo - np.mean(campo) 
        flut_vel = vel - np.mean(vel) 
        flut_eqw = equi - np.mean(equi) 
        #flut_eqw_300 = equi_300 - np.mean(equi_300) 
        #flut_eqw_240 = equi_240 - np.mean(equi_240) 
        #flut_eqw_600 = equi_600 - np.mean(equi_600) 

        Rk_array[f,:] = compute_Rk(flut_tau, flut_vel, k_array, nx,nz)
        Rk_ewm[f,:] = compute_Rk(flut_eqw, flut_vel, k_array, nx,nz)
        #Rk_ewm_t300[f,:] = compute_Rk(flut_eqw_300, flut_vel, k_array, nx,nz)
        #Rk_ewm_t240[f,:] = compute_Rk(flut_eqw_240, flut_vel, k_array, nx,nz)
        #Rk_ewm_t600[f,:] = compute_Rk(flut_eqw_600, flut_vel, k_array, nx,nz)
        f = f + 1

#f = 0
#for (k,ku) in zip(model.keys(), u60.keys()):
#    campo = model[k]
#    vel = u60[ku]
#    flut_model = campo - np.mean(campo) 
#    flut_vel = vel - np.mean(vel)
#
#    Rk_model[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
#    f = f + 1
#
#f = 0
#for (k,ku) in zip(model1.keys(), u60.keys()):
#    campo = model1[k]
#    vel = u60[ku]
#    flut_model = campo - np.mean(campo) 
#    flut_vel = vel - np.mean(vel)
#
#    Rk_model1[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
#    f = f + 1
#
#f = 0
#for (k,ku) in zip(model2.keys(), u60.keys()):
#    campo = model2[k]
#    vel = u60[ku]
#    flut_model = campo - np.mean(campo) 
#    flut_vel = vel - np.mean(vel)
#
#    Rk_model2[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
#    f = f + 1
#
#f = 0
#for (k,ku) in zip(model3.keys(), u60.keys()):
#    campo = model3[k]
#    vel = u60[ku]
#    flut_model = campo - np.mean(campo) 
#    flut_vel = vel - np.mean(vel)
#
#    Rk_model3[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
#    f = f + 1
#
#f = 0
#for (k,ku) in zip(model4.keys(), u60.keys()):
#    campo = model4[k]
#    vel = u60[ku]
#    flut_model = campo - np.mean(campo) 
#    flut_vel = vel - np.mean(vel)
#
#    Rk_model4[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
#    f = f + 1
#
#f = 0
#for (k,ku) in zip(model_unc.keys(), u60.keys()):
#    campo = model_unc[k]
#    vel = u60[ku]
#    flut_model = campo - np.mean(campo) 
#    flut_vel = vel - np.mean(vel)
#
#    Rk_uncond[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
#    f = f + 1
#
#f = 0
#for (k,ku) in zip(linear.keys(), u60.keys()):
#    campo = linear[k]
#    vel = u60[ku]
#    flut_model = campo - np.mean(campo) 
#    flut_vel = vel - np.mean(vel)
#
#    Rk_linear[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
#    f = f + 1
#    #Rk_list = []
#    #for k in k_array:
#    #    pi_prod = 0
#    #    for m in range(nz):
#    #        for i in range(nx):
#    #            j = int(i + k)
#    #            if j < 0:
#    #                j = int(j + nx)
#    #            elif j >= nx:
#    #                j = int(j - nx)   #gestisco periodicità
#    #    
#    #            pi = campo[i,m] * vel[j,m]
#    #            pi_prod = pi_prod + pi
#
#
#    #    Ck = pi_prod / N
#    #    Rk = Ck / (var_tau_all * var_vel_all)
#    #    Rk_list.append(Rk)
#    #Rk_list = np.array(Rk_list)
#    #Rk_array[f,:] = Rk_list
#    #f = f + 1

Rk_mean = np.mean(Rk_array, axis=0)
idx_0 = np.abs(x_dimension -0).argmin()
print('corr dns 0 = {}'.format(Rk_mean[idx_0]))
print('Rk_mean = {}'.format(Rk_mean))
Rk_mean_ewm = np.mean(Rk_ewm, axis=0)
#Rk_mean_ewm_t300 = np.mean(Rk_ewm_t300, axis=0)
#Rk_mean_ewm_t240 = np.mean(Rk_ewm_t240, axis=0)
#Rk_mean_ewm_t600 = np.mean(Rk_ewm_t600, axis=0)
#Rk_mean_model = np.mean(Rk_model, axis=0)
#Rk_mean_model1 = np.mean(Rk_model1, axis=0)
#Rk_mean_model2 = np.mean(Rk_model2, axis=0)
#Rk_mean_model3 = np.mean(Rk_model3, axis=0)
#Rk_mean_model4 = np.mean(Rk_model4, axis=0)
#Rk_mean_uncond = np.mean(Rk_uncond, axis=0)
#Rk_mean_linear = np.mean(Rk_linear, axis=0)
fig,ax = plt.subplots(figsize=(12,12))
ax.plot(x_dimension, Rk_mean, color='black',label='dns correlation y+=120')
ax.plot(x_dimension, Rk_mean_ewm, label='ewm correlation y+=120')
#ax.plot(x_dimension, Rk_mean_ewm_t300, label='ewm t300 correlation y+=60')
#ax.plot(x_dimension, Rk_mean_ewm_t240, label='ewm t240 correlation y+=60')
#ax.plot(x_dimension, Rk_mean_ewm_t600, label='ewm t600 correlation y+=60')
#ax.plot(x_dimension, Rk_mean_model, label='DM correlation y+=60 t600 r600')
#ax.plot(x_dimension, Rk_mean_model1, label='DM correlation y+=60 t300 r300')
#ax.plot(x_dimension, Rk_mean_model2, label='DM correlation y+=60 t300 r180')
#ax.plot(x_dimension, Rk_mean_model3, label='DM correlation y+=60 t240 r30')
#ax.plot(x_dimension, Rk_mean_model4, label='DM correlation y+=60 control error')
#ax.plot(x_dimension, Rk_mean_uncond, label='DM correlation y+=60 uncond')
#ax.plot(x_dimension, Rk_mean_linear, label='DM correlation y+=60 a*EWM + (1-a)*UNCOND')

ax.set_xlim(-0.5,0.5)
plt.legend()
plt.savefig('cross_correlation_alpha_central_y120.png', dpi=300)
plt.close()
