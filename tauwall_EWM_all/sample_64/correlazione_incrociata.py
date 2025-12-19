import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dns = np.load('tauwall_target_64.npy')
u10 = np.load('slice_u10_64.npy')
u15 = np.load('slice_u15_64.npy')
u20 = np.load('slice_u20_64.npy')
u30 = np.load('slice_u30_64.npy')
u40 = np.load('slice_u40_64.npy')
u50 = np.load('slice_u50_64.npy')
u60 = np.load('slice_u60_64.npy')
u70 = np.load('slice_u70_64.npy')
u80 = np.load('slice_u80_64.npy')
u90 = np.load('slice_u90_64.npy')
u100 = np.load('slice_u100_64.npy')
u110 = np.load('slice_u110_64.npy')
u120 = np.load('slice_u120_64.npy')
u130 = np.load('slice_u130_64.npy')
u140 = np.load('slice_u140_64.npy')
u150 = np.load('slice_u150_64.npy')
u160 = np.load('slice_u160_64.npy')
u170 = np.load('slice_u170_64.npy')
u180 = np.load('slice_u180_64.npy')
u190 = np.load('slice_u190_64.npy')
u200 = np.load('slice_u200_64.npy')

nx = 1216
nz = 640
delta = 0.5
lx = 5
lz = 1.5
dx = lx / (nx-1)
dz = lz / (nz-1)


#coefficiente k che stabilisce la grandezza della finestra per la corss correlation
kmax = 1.5 * delta // dx
delta_x_max = kmax * dx
if delta_x_max > lx / 2:
    kmax = nx // 2

kmax=64
print(kmax)
k_array = np.linspace(-kmax,kmax,2*64+1)

x_dimension = []
for k in k_array:
    a = k* dx / delta
    x_dimension.append(a)
x_dimension = np.array(x_dimension)
print(x_dimension)
print(x_dimension.shape)
Rk_dns = np.zeros((67*190, 1)) #k_array.shape[0]))
Rk_u10 = np.zeros((67*190, 1)) #k_array.shape[0]))
Rk_u15 = np.zeros((67*190, 1)) #k_array.shape[0]))
Rk_u20 = np.zeros((67*190, 1)) #k_array.shape[0]))
Rk_u30 = np.zeros((67*190, 1)) #k_array.shape[0]))
Rk_u40 = np.zeros((67*190, 1)) #k_array.shape[0]))
Rk_u50 = np.zeros((67*190, 1)) #k_array.shape[0]))
Rk_u60 = np.zeros((67*190, 1)) #k_array.shape[0]))
Rk_u70 = np.zeros((67*190, 1)) #k_array.shape[0]))
Rk_u80 = np.zeros((67*190, 1)) #k_array.shape[0]))
Rk_u90 = np.zeros((67*190, 1)) #k_array.shape[0]))
Rk_u100 = np.zeros((67*190, 1))#k_array.shape[0]))
Rk_u110 = np.zeros((67*190, 1))#k_array.shape[0]))
Rk_u120 = np.zeros((67*190, 1))#k_array.shape[0]))
Rk_u130 = np.zeros((67*190, 1))#k_array.shape[0]))
Rk_u140 = np.zeros((67*190, 1))#k_array.shape[0]))
Rk_u150 = np.zeros((67*190, 1))#k_array.shape[0]))
Rk_u160 = np.zeros((67*190, 1))#k_array.shape[0]))
Rk_u170 = np.zeros((67*190, 1))#k_array.shape[0]))
Rk_u180 = np.zeros((67*190, 1))#k_array.shape[0]))
Rk_u190 = np.zeros((67*190, 1))#k_array.shape[0]))
Rk_u200 = np.zeros((67*190, 1))#k_array.shape[0]))

from numba import njit,prange
nx = 64
nz = 64

@njit(parallel=True, fastmath=True)
#def compute_Rk(fluct_tau, fluct_vel, k_array, nx,nz):
#    Nk = k_array.size
#    N = nx * nz
#    sigma_tau = np.sqrt(np.mean(fluct_tau**2))
#    sigma_vel = np.sqrt(np.mean(fluct_vel**2))
#    denom = sigma_tau * sigma_vel
#    out = np.empty(Nk, dtype=np.float64)
#
#    for kk in prange(Nk):
#        k = int(k_array[kk])
#        prod = 0.0
#        f = 0  
#        for m in range(nz):
#            for i in range(nx):
#                j = (i-k)
#                if j > 0 and j < nx :
#                prod = prod + fluct_tau[i,m] * fluct_vel[i,m]
#                f = f + 1
#        Ck = prod / f
#        out[kk] = Ck / denom if denom > 0.0 else 0.0
#    return out
def compute_Rk(fluct_tau, fluct_vel, k_array, nx,nz):
    sigma_tau = np.sqrt(np.mean(fluct_tau**2))
    sigma_vel = np.sqrt(np.mean(fluct_vel**2))
    denom = sigma_tau * sigma_vel
    N = nx * nz
    out = np.zeros((nx,nz), dtype=np.float64)

    for m in range(nz):
        for i in range(nx):
            prod = fluct_tau[i,m]*fluct_vel[i,m]
            out[i,m] = prod

    out_mean = np.mean(out)
    out_mean = out_mean / denom

    return out_mean



f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel10 = u10[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel10 = vel10 - np.mean(vel10)

        Rk_u10[f,:] = compute_Rk(flut_tau, flut_vel10, k_array, nx,nz)
        f = f + 1

f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel15 = u15[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel15 = vel15 - np.mean(vel15)

        Rk_u15[f,:] = compute_Rk(flut_tau, flut_vel15, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel20 = u20[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel20 = vel20 - np.mean(vel20)

        Rk_u20[f,:] = compute_Rk(flut_tau, flut_vel20, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel30 = u30[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel30 = vel30 - np.mean(vel30)

        Rk_u30[f,:] = compute_Rk(flut_tau, flut_vel30, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel40 = u40[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel40 = vel40 - np.mean(vel40)

        Rk_u40[f,:] = compute_Rk(flut_tau, flut_vel40, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel50 = u50[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel50 = vel50 - np.mean(vel50)

        Rk_u50[f,:] = compute_Rk(flut_tau, flut_vel50, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel60 = u60[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel60 = vel60 - np.mean(vel60)

        Rk_u60[f,:] = compute_Rk(flut_tau, flut_vel60, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel70 = u70[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel70 = vel70 - np.mean(vel70)

        Rk_u70[f,:] = compute_Rk(flut_tau, flut_vel70, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel80 = u80[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel80 = vel80 - np.mean(vel80)

        Rk_u80[f,:] = compute_Rk(flut_tau, flut_vel80, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel90 = u90[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel90 = vel90 - np.mean(vel90)

        Rk_u90[f,:] = compute_Rk(flut_tau, flut_vel90, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel100 = u100[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel100 = vel100 - np.mean(vel100)

        Rk_u100[f,:] = compute_Rk(flut_tau, flut_vel100, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel110 = u110[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel110 = vel110 - np.mean(vel110)

        Rk_u110[f,:] = compute_Rk(flut_tau, flut_vel110, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel120 = u120[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel120 = vel120 - np.mean(vel120)

        Rk_u120[f,:] = compute_Rk(flut_tau, flut_vel120, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel130 = u130[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel130 = vel130 - np.mean(vel130)

        Rk_u130[f,:] = compute_Rk(flut_tau, flut_vel130, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel140 = u140[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel140 = vel140 - np.mean(vel140)

        Rk_u140[f,:] = compute_Rk(flut_tau, flut_vel140, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel150 = u150[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel150 = vel150 - np.mean(vel150)

        Rk_u150[f,:] = compute_Rk(flut_tau, flut_vel150, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel160 = u160[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel160 = vel160 - np.mean(vel160)

        Rk_u160[f,:] = compute_Rk(flut_tau, flut_vel160, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel170 = u170[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel170 = vel170 - np.mean(vel170)

        Rk_u170[f,:] = compute_Rk(flut_tau, flut_vel170, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel180 = u180[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel180 = vel180 - np.mean(vel180)

        Rk_u180[f,:] = compute_Rk(flut_tau, flut_vel180, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel190 = u190[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel190 = vel190 - np.mean(vel190)

        Rk_u190[f,:] = compute_Rk(flut_tau, flut_vel190, k_array, nx,nz)
        f = f + 1
f = 0
for i in range(u10.shape[0]):
    for j in range(u10.shape[1]):
        campo = dns[i][j]
        vel200 = u200[i][j]
    
        flut_tau = campo - np.mean(campo)
        flut_vel200 = vel200 - np.mean(vel200)

        Rk_u200[f,:] = compute_Rk(flut_tau, flut_vel200, k_array, nx,nz)
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

Rk_u10_mean = np.mean(Rk_u10, axis=0)
Rk_u15_mean = np.mean(Rk_u15, axis=0)
Rk_u20_mean = np.mean(Rk_u20, axis=0)
Rk_u30_mean = np.mean(Rk_u30, axis=0)
Rk_u40_mean = np.mean(Rk_u40, axis=0)
Rk_u50_mean = np.mean(Rk_u50, axis=0)
Rk_u60_mean = np.mean(Rk_u60, axis=0)
Rk_u70_mean = np.mean(Rk_u70, axis=0)
Rk_u80_mean = np.mean(Rk_u80, axis=0)
Rk_u90_mean = np.mean(Rk_u90, axis=0)
Rk_u100_mean = np.mean(Rk_u100, axis=0)
Rk_u110_mean = np.mean(Rk_u110, axis=0)
Rk_u120_mean = np.mean(Rk_u120, axis=0)
Rk_u130_mean = np.mean(Rk_u130, axis=0)
Rk_u140_mean = np.mean(Rk_u140, axis=0)
Rk_u150_mean = np.mean(Rk_u150, axis=0)
Rk_u160_mean = np.mean(Rk_u160, axis=0)
Rk_u170_mean = np.mean(Rk_u170, axis=0)
Rk_u180_mean = np.mean(Rk_u180, axis=0)
Rk_u190_mean = np.mean(Rk_u190, axis=0)
Rk_u200_mean = np.mean(Rk_u200, axis=0)
np.save('Rk_u10.npy',Rk_u10_mean)
np.save('Rk_u15.npy',Rk_u15_mean)
np.save('Rk_u20.npy',Rk_u20_mean)
np.save('Rk_u30.npy',Rk_u30_mean)
np.save('Rk_u40.npy',Rk_u40_mean)
np.save('Rk_u50.npy',Rk_u50_mean)
np.save('Rk_u60.npy',Rk_u60_mean)
np.save('Rk_u70.npy',Rk_u70_mean)
np.save('Rk_u80.npy',Rk_u80_mean)
np.save('Rk_u90.npy',Rk_u90_mean)
np.save('Rk_u100.npy',Rk_u100_mean)
np.save('Rk_u110.npy',Rk_u110_mean)
np.save('Rk_u120.npy',Rk_u120_mean)
np.save('Rk_u130.npy',Rk_u130_mean)
np.save('Rk_u140.npy',Rk_u140_mean)
np.save('Rk_u150.npy',Rk_u150_mean)
np.save('Rk_u160.npy',Rk_u160_mean)
np.save('Rk_u170.npy',Rk_u170_mean)
np.save('Rk_u180.npy',Rk_u180_mean)
np.save('Rk_u190.npy',Rk_u190_mean)
np.save('Rk_u200.npy',Rk_u200_mean)
fig,ax = plt.subplots(figsize=(12,12))
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u10_mean, label='correlation y+=10')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u15_mean, label='correlation y+=15')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u20_mean, label='correlation y+=20')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u30_mean, label='correlation y+=30')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u40_mean, label='correlation y+=40')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u50_mean, label='correlation y+=50')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u60_mean, label='correlation y+=60')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u70_mean, label='correlation y+=70')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u80_mean, label='correlation y+=80')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u90_mean, label='correlation y+=90')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u100_mean, label='correlation y+=100')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u110_mean, label='correlation y+=110')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u120_mean, label='correlation y+=120')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u130_mean, label='correlation y+=130')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u140_mean, label='correlation y+=140')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u150_mean, label='correlation y+=150')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u160_mean, label='correlation y+=160')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u170_mean, label='correlation y+=170')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u180_mean, label='correlation y+=180')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u190_mean, label='correlation y+=190')
ax.scatter(x_dimension[np.abs(x_dimension - 0).argmin()], Rk_u200_mean, label='correlation y+=200')
ax.set_xlim(-1.5,1.5)
plt.legend()
plt.savefig('cross_correlation_shift_in_avanti_all_match_central.png', dpi=300)
plt.close()
