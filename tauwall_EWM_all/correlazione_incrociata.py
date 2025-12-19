import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dns = np.load('tauwall_target.npz')
ewm = np.load('EWM_slice_match/tauwall_EWM60.npz')
u60 = np.load('slice_u60.npz')

nx = 1216
nz = 640
delta = 0.5
lx = 5
lz = 1.5
dx = lx / (nx-1)
dz = lz / (nz-1)


#dns_all = []
#ewm_all = []
#u60_all = []
#for (kd,ke,ku) in zip(dns.keys(), ewm.keys(), u60.keys()):
#    campo_d = dns[kd].ravel()
#    campo_e = ewm[ke].ravel()
#    campo_u = u60[ku].ravel()
#    dns_all.append(campo_d)
#    ewm_all.append(campo_e)
#    u60_all.append(campo_u)
#
#dns_all = np.array(dns_all)
#ewm_all = np.array(ewm_all)
#u60_all = np.array(u60_all)
#
#mean_dns = np.mean(dns_all)
#mean_ewm = np.mean(ewm_all)
#mean_u60 = np.mean(u60_all)
#
#utau_mean_dns = np.sqrt(mean_dns)
#utau_mean_ewm = np.sqrt(mean_ewm)
#
#
#flut_dns = (dns_all - mean_dns) / mean_dns
#flut_ewm = (ewm_all - mean_ewm) / mean_ewm
#flut_u60 = (u60_all - mean_u60) / utau_mean_dns


#coefficiente k che stabilisce la grandezza della finestra per la corss correlation
kmax = 1.5 * delta // dx
delta_x_max = kmax * dx
if delta_x_max > lx / 2:
    kmax = nx // 2


print(kmax)
k_array = np.linspace(-kmax,kmax,2*182+1)

x_dimension = []
for k in k_array:
    a = k* dx / delta
    x_dimension.append(a)
x_dimension = np.array(x_dimension)
print(x_dimension)
print(x_dimension.shape)
Rk_array = np.zeros((67,k_array.shape[0]))
Rk_ewm = np.zeros((67,k_array.shape[0]))
Rk_dns_ewm = np.zeros((67,k_array.shape[0]))

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
        for m in range(nz):
            for i in range(nx):
                j = (i-k) % nx
                prod = prod + fluct_tau[i,m] * fluct_vel[j,m]
        Ck = prod / N
        out[kk] = Ck / denom if denom > 0.0 else 0.0
    return out
                
f = 0
for (kd,ku,ke) in zip(dns.keys(), u60.keys(), ewm.keys()):
    campo = dns[kd]
    vel = u60[ku]
    equi = ewm[ke]
    flut_tau = campo - np.mean(campo)
    flut_vel = vel - np.mean(vel)
    flut_eqw = equi - np.mean(equi)

    Rk_array[f,:] = compute_Rk(flut_tau, flut_vel, k_array, nx,nz)
    #Rk_ewm[f,:] = compute_Rk(flut_eqw, flut_vel, k_array, nx,nz)
    Rk_dns_ewm[f,:] = compute_Rk(flut_tau, flut_eqw, k_array, nx,nz)
    f = f + 1

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
#Rk_mean_ewm = np.mean(Rk_ewm, axis=0)
Rk_mean_dns_ewm = np.mean(Rk_dns_ewm, axis=0)
idx0 = np.abs(x_dimension - 0).argmin()
fig,ax = plt.subplots()
ax.set_title('corr in 0 : dns-vel = {}, dns-ewm = {}'.format(Rk_mean[idx0], Rk_mean_dns_ewm[idx0]))
ax.plot(x_dimension, Rk_mean, color='black',label='dns correlation con y+=60')
#ax.plot(x_dimension, Rk_mean_ewm, label='ewm correlation y+=60')
ax.plot(x_dimension, Rk_mean_dns_ewm, label='ewm correlation con EWM')
ax.set_xlim(-1.5,1.5)
ax.set_ylim(0,1.05)
plt.legend()
plt.savefig('cross_correlation_shift_in_avanti.png', dpi=300)
plt.close()
