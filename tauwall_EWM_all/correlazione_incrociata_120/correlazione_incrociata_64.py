import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dns = np.load('tauwall_dns_120.npz')
ewm = np.load('tauwall_EWM120_model.npz')
u60 = np.load('sample_u120_model.npz')
model = np.load('tauwall_ML_model_t820_r102_120.npz')
model1 = np.load('tauwall_ML_model_t900_r100_120.npz')
model2 = np.load('tauwall_ML_model_t880_r110_120.npz')
#model2 = np.load('tauwall_ML_model_t240_r240.npz')
#model3 = np.load('tauwall_ML_model_t240_r30.npz')
#model4 = np.load('tauwall_ML_model_customized.npz')
#model5 = np.load('tauwall_ML_model_t720_r90.npz')
#model6 = np.load('tauwall_ML_model_t1000_r125.npz')
#model7 = np.load('tauwall_ML_model_t880_r110.npz')
#model8 = np.load('tauwall_ML_model_t840_r105.npz')
#model9 = np.load('tauwall_ML_model_t820_r102.npz')
#model10 = np.load('tauwall_ML_model_t800_r100.npz')
#model_unc = np.load('tauwall_unconditional.npz')
#
##model_unc = {}
##for i in range(uncond.shape[0]):
##    key = '{}'.format(i)
##    model_unc[key] = uncond[i][0]
#
#
#alpha = 0.1388384832398331
#alpha_max = 0.25370291
#alpha_max = 0.5
#linear = {}
#for (i,k) in zip(model_unc.keys(), ewm.keys()):
#    campo_e = ewm[k]
#    model_u = model_unc[i]
#    campo = alpha * campo_e + (1 - alpha) * model_u
#    linear[k] = campo


nx = 1216
nz = 640
delta = 0.5
lx = 5
lz = 1.5
dx = lx / (nx-1)
dz = lz / (nz-1)

len_dns = len(dns.keys())
len_ewm = len(ewm.keys())
len_u60 = len(u60.keys())
len_model = len(model.keys())
len_model1 = len(model1.keys())
len_model2 = len(model2.keys())
#len_model3 = len(model3.keys())
#len_model4 = len(model4.keys())
#len_model5 = len(model5.keys())
#len_model6 = len(model6.keys())
#len_model7 = len(model7.keys())
#len_model8 = len(model8.keys())
#len_model9 = len(model9.keys())
#len_model10 = len(model10.keys())
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
#attenuation_720 = 0.07231633
#noise_level_720 = 0.99738175

data_mean= 0.002453068295866685 #statistiche del training set 
data_std= 0.0010229427221970142


#ewm_t300 = {}
#ewm_t240 = {}
#ewm_t600 = {}
#ewm_t720 = {}
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
#    campo_720 = campo * attenuation_720 + noise_level_720 * noise
#    campo_720 = campo_720 * data_std + data_mean
#    ewm_t720[k] = campo_720


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
Rk_array = np.zeros((len_dns,k_array.shape[0]))
Rk_ewm = np.zeros((len_ewm,k_array.shape[0]))
#Rk_ewm_t300 = np.zeros((len_ewm,k_array.shape[0]))
#Rk_ewm_t240 = np.zeros((len_ewm,k_array.shape[0]))
#Rk_ewm_t600 = np.zeros((len_ewm,k_array.shape[0]))
#Rk_ewm_t720 = np.zeros((len_ewm,k_array.shape[0]))
Rk_model = np.zeros((len_model,k_array.shape[0]))
Rk_model1 = np.zeros((len_model1,k_array.shape[0]))
Rk_model2 = np.zeros((len_model2,k_array.shape[0]))
#Rk_model3 = np.zeros((len_model3,k_array.shape[0]))
#Rk_model4 = np.zeros((len_model4,k_array.shape[0]))
#Rk_model5 = np.zeros((len_model5,k_array.shape[0]))
#Rk_model6 = np.zeros((len_model6,k_array.shape[0]))
#Rk_model7 = np.zeros((len_model7,k_array.shape[0]))
#Rk_model8 = np.zeros((len_model8,k_array.shape[0]))
#Rk_model9 = np.zeros((len_model9,k_array.shape[0]))
#Rk_model10 = np.zeros((len_model10,k_array.shape[0]))
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
for (kd,ku,ke) in zip(dns.keys(), u60.keys(), ewm.keys()):
    campo = dns[kd]
    vel = u60[ku]
    equi = ewm[ke]
    #equi_300 = ewm_t300[ke]
    #equi_240 = ewm_t240[ke]
    #equi_600 = ewm_t600[ke]
    #equi_720 = ewm_t720[ke]

    flut_tau = campo - np.mean(campo) 
    flut_vel = vel - np.mean(vel) 
    flut_eqw = equi - np.mean(equi) 
    #flut_eqw_300 = equi_300 - np.mean(equi_300) 
    #flut_eqw_240 = equi_240 - np.mean(equi_240) 
    #flut_eqw_600 = equi_600 - np.mean(equi_600) 
    #flut_eqw_720 = equi_720 - np.mean(equi_720) 

    Rk_array[f,:] = compute_Rk(flut_tau, flut_vel, k_array, nx,nz)
    Rk_ewm[f,:] = compute_Rk(flut_eqw, flut_vel, k_array, nx,nz)
    #Rk_ewm_t300[f,:] = compute_Rk(flut_eqw_300, flut_vel, k_array, nx,nz)
    #Rk_ewm_t240[f,:] = compute_Rk(flut_eqw_240, flut_vel, k_array, nx,nz)
    #Rk_ewm_t600[f,:] = compute_Rk(flut_eqw_600, flut_vel, k_array, nx,nz)
    #Rk_ewm_t720[f,:] = compute_Rk(flut_eqw_720, flut_vel, k_array, nx,nz)
    f = f + 1

f = 0
for (k,ku) in zip(model.keys(), u60.keys()):
    campo = model[k]
    vel = u60[ku]
    flut_model = campo - np.mean(campo) 
    flut_vel = vel - np.mean(vel)

    Rk_model[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
    f = f + 1

f = 0
for (k,ku) in zip(model1.keys(), u60.keys()):
    campo = model1[k]
    vel = u60[ku]
    flut_model = campo - np.mean(campo) 
    flut_vel = vel - np.mean(vel)

    Rk_model1[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
    f = f + 1
#
f = 0
for (k,ku) in zip(model2.keys(), u60.keys()):
    campo = model2[k]
    vel = u60[ku]
    flut_model = campo - np.mean(campo) 
    flut_vel = vel - np.mean(vel)

    Rk_model2[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
    f = f + 1
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
#for (k,ku) in zip(model5.keys(), u60.keys()):
#    campo = model5[k]
#    vel = u60[ku]
#    flut_model = campo - np.mean(campo) 
#    flut_vel = vel - np.mean(vel)
#
#    Rk_model5[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
#    f = f + 1
#
#f = 0
#for (k,ku) in zip(model6.keys(), u60.keys()):
#    campo = model6[k]
#    vel = u60[ku]
#    flut_model = campo - np.mean(campo) 
#    flut_vel = vel - np.mean(vel)
#
#    Rk_model6[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
#    f = f + 1
#
#f = 0
#for (k,ku) in zip(model7.keys(), u60.keys()):
#    campo = model7[k]
#    vel = u60[ku]
#    flut_model = campo - np.mean(campo) 
#    flut_vel = vel - np.mean(vel)
#
#    Rk_model7[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
#    f = f + 1
#
#f = 0
#for (k,ku) in zip(model8.keys(), u60.keys()):
#    campo = model8[k]
#    vel = u60[ku]
#    flut_model = campo - np.mean(campo) 
#    flut_vel = vel - np.mean(vel)
#
#    Rk_model8[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
#    f = f + 1
#
#f = 0
#for (k,ku) in zip(model9.keys(), u60.keys()):
#    campo = model9[k]
#    vel = u60[ku]
#    flut_model = campo - np.mean(campo) 
#    flut_vel = vel - np.mean(vel)
#
#    Rk_model9[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
#    f = f + 1
#
#f = 0
#for (k,ku) in zip(model10.keys(), u60.keys()):
#    campo = model10[k]
#    vel = u60[ku]
#    flut_model = campo - np.mean(campo) 
#    flut_vel = vel - np.mean(vel)
#
#    Rk_model10[f,:] = compute_Rk(flut_model, flut_vel, k_array, nx,nz)
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
    #Rk_list = []
    #for k in k_array:
    #    pi_prod = 0
    #    for m in range(nz):
    #        for i in range(nx):
    #            j = int(i + k)
    #            if j < 0:
    #                j = int(j + nx)
    #            elif j >= nx:
    #                j = int(j - nx)   #gestisco periodicità
    #    
    #            pi = campo[i,m] * vel[j,m]
    #            pi_prod = pi_prod + pi


    #    Ck = pi_prod / N
    #    Rk = Ck / (var_tau_all * var_vel_all)
    #    Rk_list.append(Rk)
    #Rk_list = np.array(Rk_list)
    #Rk_array[f,:] = Rk_list
    #f = f + 1

Rk_mean = np.mean(Rk_array, axis=0)
np.save('Rk_dns_120.npy',Rk_mean)
idx_0 = np.abs(x_dimension -0).argmin()
print('corr dns 0 = {}'.format(Rk_mean[idx_0]))
print('Rk_mean = {}'.format(Rk_mean))
Rk_mean_ewm = np.mean(Rk_ewm, axis=0)
np.save('Rk_ewm_120.npy',Rk_mean_ewm)
#Rk_mean_ewm_t300 = np.mean(Rk_ewm_t300, axis=0)
#np.save('Rk_ewm_t300.npy',Rk_mean_ewm_t300)
#Rk_mean_ewm_t240 = np.mean(Rk_ewm_t240, axis=0)
#np.save('Rk_ewm_t240.npy',Rk_mean_ewm_t240)
#Rk_mean_ewm_t600 = np.mean(Rk_ewm_t600, axis=0)
#np.save('Rk_ewm_t600.npy',Rk_mean_ewm_t600)
#Rk_mean_ewm_t720 = np.mean(Rk_ewm_t720, axis=0)
#np.save('Rk_ewm_t720.npy',Rk_mean_ewm_t720)
Rk_mean_model = np.mean(Rk_model, axis=0)
np.save('Rk_DM_t820_r102.npy', Rk_mean_model)
Rk_mean_model1 = np.mean(Rk_model1, axis=0)
np.save('Rk_DM_t900_r100.npy', Rk_mean_model1)
Rk_mean_model2 = np.mean(Rk_model2, axis=0)
np.save('Rk_DM_t880_r110.npy', Rk_mean_model2)
#Rk_mean_model3 = np.mean(Rk_model3, axis=0)
#np.save('Rk_DM_t240_r30.npy', Rk_mean_model3)
#Rk_mean_model4 = np.mean(Rk_model4, axis=0)
#np.save('Rk_DM_control_error.npy', Rk_mean_model4)
#Rk_mean_model5 = np.mean(Rk_model5, axis=0)
#np.save('Rk_DM_t720_r90.npy', Rk_mean_model5)
#Rk_mean_model6 = np.mean(Rk_model6, axis=0)
#np.save('Rk_DM_t1000_r125.npy', Rk_mean_model6)
#Rk_mean_model7 = np.mean(Rk_model7, axis=0)
#np.save('Rk_DM_t880_r110.npy', Rk_mean_model7)
#Rk_mean_model8 = np.mean(Rk_model8, axis=0)
#np.save('Rk_DM_t840_r105.npy', Rk_mean_model8)
#Rk_mean_model9 = np.mean(Rk_model9, axis=0)
#np.save('Rk_DM_t820_r102.npy', Rk_mean_model9)
#Rk_mean_model10 = np.mean(Rk_model10, axis=0)
#np.save('Rk_DM_t800_r100.npy', Rk_mean_model10)
#exit()
#Rk_mean_uncond = np.mean(Rk_uncond, axis=0)
#np.save('Rk_DM_uncond.npy', Rk_mean_uncond)
#Rk_mean_linear = np.mean(Rk_linear, axis=0)
#np.save('Rk_DM_linear_alpha_01388.npy', Rk_mean_linear)
fig,ax = plt.subplots(figsize=(12,12))
ax.plot(x_dimension, Rk_mean, color='black',label='dns correlation y+=120')
ax.plot(x_dimension, Rk_mean_ewm, label='ewm correlation y+=120')
#ax.plot(x_dimension, Rk_mean_ewm_t300, label='ewm t300 correlation y+=60')
#ax.plot(x_dimension, Rk_mean_ewm_t240, label='ewm t240 correlation y+=60')
#ax.plot(x_dimension, Rk_mean_ewm_t600, label='ewm t600 correlation y+=60')
#ax.plot(x_dimension, Rk_mean_ewm_t720, label='ewm t720 correlation y+=60')
ax.plot(x_dimension, Rk_mean_model, label='DM correlation y+=120 t820 r102')
ax.plot(x_dimension, Rk_mean_model1, label='DM correlation y+=120 t900 r100')
ax.plot(x_dimension, Rk_mean_model2, label='DM correlation y+=120 t880 r110')
#ax.plot(x_dimension, Rk_mean_model3, label='DM correlation y+=60 t240 r30')
#ax.plot(x_dimension, Rk_mean_model4, label='DM correlation y+=60 control error')
#ax.plot(x_dimension, Rk_mean_model5, label='DM correlation y+=60 t720 r90')
#ax.plot(x_dimension, Rk_mean_model6, label='DM correlation y+=60 t1000 r125')
#ax.plot(x_dimension, Rk_mean_uncond, label='DM correlation y+=60 uncond')
#ax.plot(x_dimension, Rk_mean_linear, label='DM correlation y+=60 a*EWM + (1-a)*UNCOND')

ax.set_xlim(-0.5,0.5)
plt.legend()
plt.savefig('cross_correlation_120.png', dpi=300)
plt.close()
