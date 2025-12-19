import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt,cm
import seaborn as sns
from scipy.ndimage import gaussian_filter
from function import*
from scipy.stats import mode


#parametri
lx = 5.
nx = 1216
ly = 1.5
ny = 640
lz = 1.
nz = 512
Re_tau = 1.02003270E+003
utau_mean  = 4.97576926E-002
delta  = lz / 2
delta_v = delta / Re_tau
delta_x = lx / nx
delta_y = ly / ny
#delta_z = lz / nz
delta_x_plus = delta_x / delta_v
delta_y_plus = delta_y / delta_v
rho = 1.
mu = utau_mean * delta_v
normal_grid = np.genfromtxt('grid.out')
zc_cord = normal_grid[:,2]
zf_cord = normal_grid[:,1]

x = np.linspace(0,lx,nx)
y = np.linspace(0,ly,ny)

dzc = normal_grid[:,4]
dzf = normal_grid[:,3]

zc_cord[0] = 0
zc_cord[-1] = 1 #dimezzo spessore di griglia a parete per calolare in modo ottimale grad_dns
zc_cord_plus = np.zeros(zc_cord.shape[0])

z_plus = 0
for i in range(zc_cord.shape[0]-1):
    delta_z_plus = (zc_cord[i+1] - zc_cord[i]) / delta_v
    z_plus = z_plus + delta_z_plus
    zc_cord_plus[i+1] = z_plus

zc_cord = zc_cord[1:-1]             #tolgo i ghost visto che non sono salvati sui campi 3d
zc_cord_plus = zc_cord_plus[1:-1]
zf_cord = zf_cord[1:-1]


id1 = np.abs(zc_cord_plus - 60).argmin()  #stesso indice che dovrò prendere per stampare le slice visto che ho tolto i ghost
id2 = np.abs(zc_cord_plus - 120).argmin()
id3 = np.abs(zc_cord_plus - 30).argmin()
id4 = np.abs(zc_cord_plus - 10).argmin()
print(id1)
print(id2)
print('indice yplus 5 = {}'.format(np.abs(zc_cord_plus - 5).argmin()))
print('indice yplus 10 = {}'.format(np.abs(zc_cord_plus - 10).argmin()))
print('indice yplus 15 = {}'.format(np.abs(zc_cord_plus - 15).argmin()))
print('indice yplus 20 = {}'.format(np.abs(zc_cord_plus - 20).argmin()))
print('indice yplus 30 = {}'.format(np.abs(zc_cord_plus - 30).argmin()))
print('indice yplus 40 = {}'.format(np.abs(zc_cord_plus - 40).argmin()))
print('indice yplus 50 = {}'.format(np.abs(zc_cord_plus - 50).argmin()))
print('indice yplus 60 = {}'.format(np.abs(zc_cord_plus - 60).argmin()))
print('indice yplus 70 = {}'.format(np.abs(zc_cord_plus - 70).argmin()))
print('indice yplus 80 = {}'.format(np.abs(zc_cord_plus - 80).argmin()))
print('indice yplus 90 = {}'.format(np.abs(zc_cord_plus - 90).argmin()))
print('indice yplus 100 = {}'.format(np.abs(zc_cord_plus - 100).argmin()))
print('indice yplus 110 = {}'.format(np.abs(zc_cord_plus - 110).argmin()))
print('indice yplus 120 = {}'.format(np.abs(zc_cord_plus - 120).argmin()))
print('indice yplus 130 = {}'.format(np.abs(zc_cord_plus - 130).argmin()))
print('indice yplus 140 = {}'.format(np.abs(zc_cord_plus - 140).argmin()))
print('indice yplus 150 = {}'.format(np.abs(zc_cord_plus - 150).argmin()))
print('indice yplus 160 = {}'.format(np.abs(zc_cord_plus - 160).argmin()))
print('indice yplus 170 = {}'.format(np.abs(zc_cord_plus - 170).argmin()))
print('indice yplus 180 = {}'.format(np.abs(zc_cord_plus - 180).argmin()))
print('indice yplus 190 = {}'.format(np.abs(zc_cord_plus - 190).argmin()))
print('indice yplus 200 = {}'.format(np.abs(zc_cord_plus - 200).argmin()))

print('-------------------------------------------------------------------------------------------------')
print('mu = {}'.format(mu))
print('y di y+ = 10 : {}'.format(zc_cord[id4]))
print('y di y+ = 30 : {}'.format(zc_cord[id3]))
print('y di y+ = 60 : {}'.format(zc_cord[id1]))
print('y di y+ = 120 : {}'.format(zc_cord[id2]))
print('y_wall = {}'.format(zc_cord[0]))
print(mu)
#---------------------------------------------------------------------------------------------------
#TEST SU UNA SLICE APPARTENENTE A TEST DATASET
test_index = [67, 37, 6, 39, 9, 15, 16, 47, 49, 19, 21, 22, 60]
keys_test = []

idx10 =np.abs(zc_cord_plus - 10).argmin()
idx15 =np.abs(zc_cord_plus - 15).argmin()
idx20 =np.abs(zc_cord_plus - 20).argmin()
idx30 =np.abs(zc_cord_plus - 30).argmin()
idx40 =np.abs(zc_cord_plus - 40).argmin()
idx50 =np.abs(zc_cord_plus - 50).argmin()
idx60 =np.abs(zc_cord_plus - 60).argmin()
idx70 =np.abs(zc_cord_plus - 70).argmin()
idx80 =np.abs(zc_cord_plus - 80).argmin()
idx90 =np.abs(zc_cord_plus - 90).argmin()
idx100 =np.abs(zc_cord_plus - 100).argmin()
idx110 =np.abs(zc_cord_plus - 110).argmin()
idx120 =np.abs(zc_cord_plus - 120).argmin()
idx130 =np.abs(zc_cord_plus - 130).argmin()
idx140 =np.abs(zc_cord_plus - 140).argmin()
idx150 =np.abs(zc_cord_plus - 150).argmin()
idx160 =np.abs(zc_cord_plus - 160).argmin()
idx170 =np.abs(zc_cord_plus - 170).argmin()
idx180 =np.abs(zc_cord_plus - 180).argmin()
idx190 =np.abs(zc_cord_plus - 190).argmin()
idx200 =np.abs(zc_cord_plus - 200).argmin()



for i in test_index:
    nome = 'u_60_{}'.format(i)
    keys_test.append(nome)


import joblib
from joblib import Parallel, delayed

u60_test = dict(np.load('../slice_Retau1000_all/slice_u200.npz'))
#u_wall = dict(np.load('slice_u_wall.npz'))
#dict_tauwall_mean = dict(np.load('dict_tauwallmean.npz'))
#utau60_mean = dict(np.load('utau60_mean.npz'))
#utau120_mean = dict(np.load('utau120_mean.npz'))
#tau_wall = {}
#for key in u_wall.keys():
#    campo = u_wall[key]
#    campo = campo * mu / zc_cord[0]
#    tau_wall[key] = campo
#medie_tauwall_test = []

pred_u60_test_tauwall = {}
#pred_u60_test_utau = {}

from numba import njit, prange
@njit(parallel=True)
def compute_utau(ymatch, campo, utau, mu):
    n,m = campo.shape
    pred = np.empty((n,m))
    for i in prange(n):
        for k in range(m):
            pred[i,k] = newton_raphson(ymatch, campo[i,k], utau[i,k],mu)[0]
    return pred


uwall_EWM = {}
for keys in u60_test.keys():
    campo = u60_test[keys]
    utau_in = np.sqrt((mu / rho) * campo / zc_cord[idx200])
    pred_u60 = compute_utau(zc_cord[idx200], campo, utau_in, mu)
    #pred_u60_test_utau[keys] = pred_u60
    pred_u60 = rho * (pred_u60**2)
    #pred_u60 = pred_u60 * medie_tauwall_test[j]
    pred_u60_test_tauwall[keys] = pred_u60
    uwall_EWM[keys] = pred_u60 * zc_cord[0] / mu


np.savez('tauwall_EWM_all/EWM_slice_match/tauwall_EWM200.npz', **pred_u60_test_tauwall)
np.savez('uwall_EWM_all/uwall_EWM200.npz', **uwall_EWM)

exit()

x_less = x[::10]
y_less = y[::10]

#for (key,keys,chiave) in zip(tauwall_test.keys(),u60_test.keys(), u120_test.keys()):    
#    fig,ax = plt.subplots(3,1,figsize=(12,17))#, constrained_layout=True)
#    ax[0].axis('scaled')
#    ax[1].axis('scaled')
#    ax[2].axis('scaled')
#    
#    vmin = min(tauwall_test[key].min(), pred_u60_test[keys].min(), pred_u120_test[chiave].min())
#    vmax = max(tauwall_test[key].max(), pred_u60_test[keys].max(), pred_u120_test[chiave].max())
#    s1 = ax[0].pcolormesh(x_less[:-1],y_less,tauwall_test[key].T, vmin=vmin, vmax=vmax, cmap='viridis', shading='auto')
#    s2 = ax[1].pcolormesh(x_less[:-1],y_less,pred_u60_test[keys].T, vmin=vmin, vmax=vmax, cmap='viridis', shading='auto')
#    s3 = ax[2].pcolormesh(x_less[:-1],y_less,pred_u120_test[chiave].T, vmin=vmin, vmax=vmax, cmap='viridis', shading='auto')
#    ax[0].set_xlim(0,5)
#    ax[0].set_ylim(0,1.5)
#    ax[1].set_xlim(0,5)
#    ax[1].set_ylim(0,1.5)
#    ax[2].set_xlim(0,5)
#    ax[2].set_ylim(0,1.5)
#    #fig.colorbar(s1, ax = [ax[0], ax[1], ax[2]], orientation='vertical', shrink=0.8, pad=1)
#    #plt.tight_layout()
#    ax[0].set_title('slice target {}'.format(key))
#    ax[1].set_title('slice predict {}'.format(keys))
#    ax[2].set_title('slice predict {}'.format(chiave))
#    # Posizione manuale della colorbar (x0, y0, width, height)
#    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
#    fig.colorbar(s1, cax=cbar_ax, label='Valore')
#    plt.subplots_adjust(right=0.9)  # lascia spazio alla colorbar
#    plt.savefig('EWM_tauwall/Slice_EWM/risultati_{}.png'.format(key), dpi=300)
#    plt.show()
#    plt.close()

#PLOT DELLE SLICE

##for (key,key60,key120,chiave60, chiave120,key30, key10) in zip(tauwall_test.keys(),u60_test.keys(), u120_test.keys(), pred_errore_u60.keys(), pred_errore_u120.keys(),u30_test.keys(), u10_test.keys()):    
##    fig,ax = plt.subplots(3,3,figsize=(20,20))#, constrained_layout=True)
##    u60_test_mean = np.mean(pred_u60_test_tauwall[key60], axis=(0,1))
##    u120_test_mean = np.mean(pred_u120_test_tauwall[key120], axis=(0,1))
##    u10_test_mean = np.mean(pred_u10_test_tauwall[key10], axis=(0,1))
##    u30_test_mean = np.mean(pred_u30_test_tauwall[key30], axis=(0,1))
##    tauwall_test_mean = np.mean(tauwall_test[key], axis=(0,1))
##    fig.suptitle('test_mean = {}, pred_u60_mean = {}, pred_u120_mean = {}, pred_u30_mean = {}, pred_u10_mean = {}'.format(tauwall_test_mean, u60_test_mean, u120_test_mean, u30_test_mean, u10_test_mean))
##    ax[0,0].axis('scaled')
##    ax[1,0].axis('scaled')
##    ax[2,0].axis('scaled')
##    ax[0,1].axis('scaled')
##    ax[1,1].axis('scaled')
##    ax[2,1].axis('scaled')
##    ax[0,2].axis('scaled')
##    ax[1,2].axis('scaled')
##    ax[2,2].axis('scaled')
##    
##    vmin = 0.0004#min(tauwall_test[key].min(), pred_u60_test[key60].min(), pred_u120_test[key120].min())
##    vmax = 0.006#max(tauwall_test[key].max(), pred_u60_test[key60].max(), pred_u120_test[key120].max())
##    vmin_errore = -1#min(pred_errore_u60[chiave60].min(), pred_errore_u120[chiave120].min())
##    vmax_errore = 1#max(pred_errore_u60[chiave60].max(), pred_errore_u120[chiave120].max())
##    s1 = ax[0,0].pcolormesh(x_less[:-1],y_less,tauwall_test[key].T, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
##    s2 = ax[0,1].pcolormesh(x_less[:-1],y_less,pred_u60_test_tauwall[key60].T, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
##    s3 = ax[1,0].pcolormesh(x_less[:-1],y_less,pred_u120_test_tauwall[key120].T, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
##    e1 = ax[0,2].pcolormesh(x_less[:-1],y_less,pred_errore_u60[chiave60].T, vmin=vmin_errore, vmax=vmax_errore, cmap='coolwarm', shading='auto')
##    e2 = ax[1,1].pcolormesh(x_less[:-1],y_less,pred_errore_u120[chiave120].T, vmin=vmin_errore, vmax=vmax_errore, cmap='coolwarm', shading='auto')
##    s4 = ax[1,2].pcolormesh(x_less[:-1],y_less,pred_u30_test_tauwall[key30].T, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
##    s5 = ax[2,1].pcolormesh(x_less[:-1],y_less,pred_u10_test_tauwall[key10].T, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
##    e4 = ax[2,0].pcolormesh(x_less[:-1],y_less,pred_errore_u30[key30].T, vmin=vmin_errore, vmax=vmax_errore, cmap='coolwarm', shading='auto')
##    e5 = ax[2,2].pcolormesh(x_less[:-1],y_less,pred_errore_u10[key10].T, vmin=vmin_errore, vmax=vmax_errore, cmap='coolwarm', shading='auto')
##    ax[0,0].set_xlim(0,5)
##    ax[0,0].set_ylim(0,1.5)
##    ax[1,0].set_xlim(0,5)
##    ax[1,0].set_ylim(0,1.5)
##    ax[2,0].set_xlim(0,5)
##    ax[2,0].set_ylim(0,1.5)
##    ax[0,1].set_xlim(0,5)
##    ax[0,1].set_ylim(0,1.5)
##    ax[1,1].set_xlim(0,5)
##    ax[1,1].set_ylim(0,1.5)
##    ax[2,1].set_xlim(0,5)
##    ax[2,1].set_ylim(0,1.5)
##    ax[0,2].set_xlim(0,5)
##    ax[0,2].set_ylim(0,1.5)
##    ax[1,2].set_xlim(0,5)
##    ax[1,2].set_ylim(0,1.5)
##    ax[2,2].set_xlim(0,5)
##    ax[2,2].set_ylim(0,1.5)
##    #fig.colorbar(s1, ax = [ax[0], ax[1], ax[2]], orientation='vertical', shrink=0.8, pad=1)
##    #plt.tight_layout()
##    ax[0,0].set_title('slice target {}'.format(key))
##    ax[0,1].set_title('slice predict {}'.format(key60))
##    ax[1,0].set_title('slice predict {}'.format(key120))
##    ax[0,2].set_title('errore u60 {}'.format(chiave60))
##    ax[1,1].set_title('errore u120 {}'.format(chiave120))
##    ax[1,2].set_title('slice predict {}'.format(key30))
##    ax[2,1].set_title('slice predict {}'.format(key10))
##    ax[2,0].set_title('errore u30 {}'.format(key30))
##    ax[2,2].set_title('errore u10 {}'.format(key10))
##    # Posizione manuale della colorbar (x0, y0, width, height)
##    cbar_ax = fig.add_axes([0.006, 0.15, 0.01, 0.7])
##    cbar_ax1 = fig.add_axes([0.93, 0.3, 0.01, 0.5])
##    fig.colorbar(s1, cax=cbar_ax, label='Valore')
##    fig.colorbar(e1, cax=cbar_ax1, label='errore')
##    plt.subplots_adjust(left=0.04, right=0.93)  # lascia spazio alla colorbar
##    plt.savefig('EWM_tauwall/Slice_allmatch_correct/risultati_tauwallstat{}.png'.format(key), dpi=300)
##    #plt.show()
##    plt.close()

#
#y_hat_u60 = np.array([])
#y_hat_u120 = np.array([])
#y = np.array([])
#for (key,chiave60,chiave120) in zip(tauwall_test.keys(),pred_u60_test_tauwall.keys(), pred_u120_test_tauwall.keys()):
#    campo_predu60 = pred_u60_test_tauwall[chiave60].ravel()
#    campo_predu120 = pred_u120_test_tauwall[chiave120].ravel()
#    campo_target = tauwall_test[key].ravel()
#    y_hat_u60 = np.append(y_hat_u60, campo_predu60)
#    y_hat_u120 = np.append(y_hat_u120, campo_predu120)
#    y = np.append(y, campo_target)
#from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#mse_u60 = mean_squared_error(y , y_hat_u60 )
#r2_u60 = r2_score(y , y_hat_u60 )
#rms_u60 = np.sqrt(mse_u60)
#mse_u120 = mean_squared_error(y , y_hat_u120 )
#r2_u120 = r2_score(y , y_hat_u120 )
#rms_u120 = np.sqrt(mse_u120)
#print('prestazioni EWM da y+=60')
#print('MSE_60 = {}'.format(mse_u60))
#print('MAE_60 = {}'.format(mean_absolute_error(y , y_hat_u60)))
#print('R2_60 = {}'.format(r2_u60))
#print('RMS_60 = {}'.format(rms_u60))
#print('------------------------------------------------------------------------------')
#print('prestazioni EWM da y+=120')
#print('MSE_120 = {}'.format(mse_u120))
#print('MAE_120 = {}'.format(mean_absolute_error(y , y_hat_u120)))
#print('R2_120 = {}'.format(r2_u120))
#print('RMS_120 = {}'.format(rms_u120))
#print('------------------------------------------------------------------------------')
#errore_u60 = np.abs(y - y_hat_u60)
#errore_u120 = np.abs(y - y_hat_u120)
#print('errore minimo  u60 = {}'.format(errore_u60.min()))
#print('errore massimo u60 = {}'.format(errore_u60.max()))
#print('85esimo percentile test_set  u60 = {}'.format(np.percentile(errore_u60, 85)))
#print('------------------------------------------------------------------------------')
#print('errore minimo  u120 = {}'.format(errore_u120.min()))
#print('errore massimo u120 = {}'.format(errore_u120.max()))
#print('85esimo percentile test_set  u120 = {}'.format(np.percentile(errore_u120, 85)))

#------------------------------------------------------------------------------------------------
#Train
u60_train = dict(np.load('u60_train_utau_tauwall_stat.npz'))
v60_train = dict(np.load('v60_train_utau_tauwall_stat.npz'))
w60_train = dict(np.load('w60_train_utau_tauwall_stat.npz'))
u30_train = dict(np.load('u30_train_utau_tauwall_stat.npz'))
v30_train = dict(np.load('v30_train_utau_tauwall_stat.npz'))
w30_train = dict(np.load('w30_train_utau_tauwall_stat.npz'))
u10_train = dict(np.load('u10_train_utau_tauwall_stat.npz'))
v10_train = dict(np.load('v10_train_utau_tauwall_stat.npz'))
w10_train = dict(np.load('w10_train_utau_tauwall_stat.npz'))
u120_train = dict(np.load('u120_train_utau_tauwall_stat.npz'))
v120_train = dict(np.load('v120_train_utau_tauwall_stat.npz'))
w120_train = dict(np.load('w120_train_utau_tauwall_stat.npz'))
tauwall_train = dict(np.load('tauwall_train_utau_tauwall_stat.npz'))

#for key in tauwall_train.keys():
#     tauwall_train[key] = tauwall_train[key] * tauwall_mean

for (key60, key120, key30, key10) in zip(u60_train.keys(), u120_train.keys(), u30_train.keys(), u10_train.keys()):
   # if key == keys:
    u60_train[key60] = u60_train[key60] * utau_mean
    u120_train[key120] = u120_train[key120] * utau_mean
    u30_train[key30] = u30_train[key30] * utau_mean
    u10_train[key10] = u10_train[key10] * utau_mean

pred_u60_train_tauwall = {}
pred_u60_train_utau = {}
pred_u30_train_tauwall = {}
pred_u30_train_utau = {}
pred_u10_train_tauwall = {}
pred_u10_train_utau = {}
for (keys,key30,key10) in zip(u60_train.keys(),u30_train.keys(),u10_train.keys()):
    campo = u60_train[keys]
    campo30 = u30_train[key30]
    campo10 = u10_train[key10]
    utau_in = np.sqrt((mu / rho) * campo / zc_cord[id1])
    utau_in30 = np.sqrt((mu / rho) * campo30 / zc_cord[id3])
    utau_in10 = np.sqrt((mu / rho) * campo10 / zc_cord[id4])
    pred_u60 = np.zeros((campo.shape[0], campo.shape[1]))
    pred_u30 = np.zeros((campo.shape[0], campo.shape[1]))
    pred_u10 = np.zeros((campo.shape[0], campo.shape[1]))
    for i in range(campo.shape[0]):
        for k in range(campo.shape[1]):
            pred_u60[i,k] = newton_raphson(zc_cord[id1], campo[i,k], utau_in[i,k], mu)[0]
            pred_u30[i,k] = newton_raphson(zc_cord[id3], campo30[i,k], utau_in30[i,k], mu)[0]
            pred_u10[i,k] = newton_raphson(zc_cord[id4], campo10[i,k], utau_in10[i,k], mu)[0]
    pred_u60_train_utau[keys] = pred_u60
    pred_u60 = rho * (pred_u60**2)
    #pred_u60 = pred_u60 * medie_tauwall_test[j]
    pred_u60_train_tauwall[keys] = pred_u60
    pred_u30_train_utau[key30] = pred_u30
    pred_u30 = rho * (pred_u30**2)
    pred_u30_train_tauwall[key30] = pred_u30
    pred_u10_train_utau[key10] = pred_u10
    pred_u10 = rho * (pred_u10**2)
    pred_u10_train_tauwall[key10] = pred_u10
    
pred_u120_train_tauwall = {}
pred_u120_train_utau = {}
for keys in u120_train.keys():
    campo = u120_train[keys]
    utau_in = np.sqrt((mu / rho) * campo / zc_cord[id1])
    pred_u120 = np.zeros((campo.shape[0], campo.shape[1]))
    for i in range(campo.shape[0]):
        for k in range(campo.shape[1]):
            pred_u120[i,k] = newton_raphson(zc_cord[id1], campo[i,k], utau_in[i,k], mu)[0]
    pred_u120_train_utau[keys] = pred_u120
    pred_u120 = rho * (pred_u120**2)
    pred_u120_train_tauwall[keys] = pred_u120

#------------------------------------------------------------------------------------------------
#valid    (sbagliato ad incollare diocan!)
#TEST SU UNA SLICE APPARTENENTE A TEST DATASET
import joblib
u60_valid = dict(np.load('u60_valid_utau_tauwall_stat.npz'))
v60_valid = dict(np.load('v60_valid_utau_tauwall_stat.npz'))
w60_valid = dict(np.load('w60_valid_utau_tauwall_stat.npz'))
u30_valid = dict(np.load('u30_valid_utau_tauwall_stat.npz'))
v30_valid = dict(np.load('v30_valid_utau_tauwall_stat.npz'))
w30_valid = dict(np.load('w30_valid_utau_tauwall_stat.npz'))
u10_valid = dict(np.load('u10_valid_utau_tauwall_stat.npz'))
v10_valid = dict(np.load('v10_valid_utau_tauwall_stat.npz'))
w10_valid = dict(np.load('w10_valid_utau_tauwall_stat.npz'))
u120_valid = dict(np.load('u120_valid_utau_tauwall_stat.npz'))
v120_valid = dict(np.load('v120_valid_utau_tauwall_stat.npz'))
w120_valid = dict(np.load('w120_valid_utau_tauwall_stat.npz'))
tauwall_valid = dict(np.load('tauwall_valid_utau_tauwall_stat.npz'))
#dict_tauwall_mean = dict(np.load('dict_tauwallmean.npz'))
#utau60_mean = dict(np.load('utau60_mean.npz'))
#utau120_mean = dict(np.load('utau120_mean.npz'))
tauwall_mean = rho * (utau_mean**2)
#medie_tauwall_test = []


#for keys in dict_tauwall_mean.keys():
###for key in tauwall_test.keys():
###    #if key == keys:
###            #medie_tauwall_test.append(dict_tauwall_mean[keys])
###     tauwall_test[key] = tauwall_test[key] * tauwall_mean


#print(medie_tauwall_test)
#print(tauwall_test['tau_wall_41'])
#print(u60_test['u_60_41'])
#print(utau60_mean['u_60_41'])

#for keys in utau60_mean.keys():
for (key60, key120, key30, key10) in zip(u60_valid.keys(), u120_valid.keys(), u30_valid.keys(), u10_valid.keys()):
   # if key == keys:
    u60_valid[key60] = u60_valid[key60] * utau_mean
    u120_valid[key120] = u120_valid[key120] * utau_mean
    u30_valid[key30] = u30_valid[key30] * utau_mean
    u10_valid[key10] = u10_valid[key10] * utau_mean

print(u60_valid.keys())
print(u30_valid.keys())
print(u10_valid.keys())

pred_u60_valid_tauwall = {}
pred_u60_valid_utau = {}
pred_u30_valid_tauwall = {}
pred_u30_valid_utau = {}
pred_u10_valid_tauwall = {}
pred_u10_valid_utau = {}
for (keys,key30,key10) in zip(u60_valid.keys(),u30_valid.keys(),u10_valid.keys()):
    campo = u60_valid[keys]
    campo30 = u30_valid[key30]
    campo10 = u10_valid[key10]
    utau_in = np.sqrt((mu / rho) * campo / zc_cord[id1])
    utau_in30 = np.sqrt((mu / rho) * campo30 / zc_cord[id3])
    utau_in10 = np.sqrt((mu / rho) * campo10 / zc_cord[id4])
    pred_u60 = np.zeros((campo.shape[0], campo.shape[1]))
    pred_u30 = np.zeros((campo.shape[0], campo.shape[1]))
    pred_u10 = np.zeros((campo.shape[0], campo.shape[1]))
    for i in range(campo.shape[0]):
        for k in range(campo.shape[1]):
            pred_u60[i,k] = newton_raphson(zc_cord[id1], campo[i,k], utau_in[i,k], mu)[0]
            pred_u30[i,k] = newton_raphson(zc_cord[id3], campo30[i,k], utau_in30[i,k], mu)[0]
            pred_u10[i,k] = newton_raphson(zc_cord[id4], campo10[i,k], utau_in10[i,k], mu)[0]    
    pred_u60_valid_utau[keys] = pred_u60
    pred_u60 = rho * (pred_u60**2)
    pred_u60_valid_tauwall[keys] = pred_u60
    pred_u30_valid_utau[key30] = pred_u30
    pred_u30 = rho * (pred_u30**2)
    pred_u30_valid_tauwall[key30] = pred_u30
    pred_u10_valid_utau[key10] = pred_u10
    pred_u10 = rho * (pred_u10**2)
    pred_u10_valid_tauwall[key10] = pred_u10
    
pred_u120_valid_utau = {}
pred_u120_valid_tauwall = {}

for keys in u120_valid.keys():
    campo = u120_valid[keys]
    utau_in = np.sqrt((mu / rho) * campo / zc_cord[id1])
    pred_u120 = np.zeros((campo.shape[0], campo.shape[1]))
    for i in range(campo.shape[0]):
        for k in range(campo.shape[1]):
            pred_u120[i,k] = newton_raphson(zc_cord[id1], campo[i,k], utau_in[i,k], mu)[0]
    pred_u120_valid_utau[keys] = pred_u120
    pred_u120 = rho * (pred_u120**2)
    pred_u120_valid_tauwall[keys] = pred_u120

u60_EWM_utau = {}
u30_EWM_utau = {}
u10_EWM_utau = {}
u120_EWM_utau = {}
tauwall_dim = {}
tauwall_dim.update(tauwall_test)
tauwall_dim.update(tauwall_train)
tauwall_dim.update(tauwall_valid)
u60_EWM_utau.update(pred_u60_test_utau)
u60_EWM_utau.update(pred_u60_train_utau)
u60_EWM_utau.update(pred_u60_valid_utau)
u30_EWM_utau.update(pred_u30_test_utau)
u30_EWM_utau.update(pred_u30_train_utau)
u30_EWM_utau.update(pred_u30_valid_utau)
u10_EWM_utau.update(pred_u10_test_utau)
u10_EWM_utau.update(pred_u10_train_utau)
u10_EWM_utau.update(pred_u10_valid_utau)
u120_EWM_utau.update(pred_u120_test_utau)
u120_EWM_utau.update(pred_u120_train_utau)
u120_EWM_utau.update(pred_u120_valid_utau)

np.savez('u60_EWM_utau_stat.npz', **u60_EWM_utau)
np.savez('u30_EWM_utau_stat.npz', **u30_EWM_utau)
np.savez('u10_EWM_utau_stat.npz', **u10_EWM_utau)
np.savez('u120_EWM_utau_stat.npz', **u120_EWM_utau)
np.savez('tauwall_dim.npz', **tauwall_dim)

print('u60_EWM keys = {}'.format(u60_EWM_utau.keys()))
print('u30_EWM keys = {}'.format(u30_EWM_utau.keys()))
print('u10_EWM keys = {}'.format(u10_EWM_utau.keys()))
exit()
#------------------------------------------------------------------------------------------------
utau_in_60 = np.sqrt((mu / rho) * u60 / zc_cord[id1])
utau_in_120 = np.sqrt((mu / rho) * u120 / zc_cord[id2])
utau_in = np.zeros((utau_in_60.shape[0],2))
utau_in[:,0] = utau_in_60
utau_in[:,1] = utau_in_120
np.save('cfd_ml_update/DATA/utau_in.npy', utau_in)
grad_wall_mean = rho  * (utau_mean**2)
target_normalized = target / grad_wall_mean
#EWM con newton raphson
utau_60 = np.zeros(u60.shape[0])
utau_120 = np.zeros(u120.shape[0])
grad_model_u60 = np.zeros(u60.shape[0])
grad_model_u120 = np.zeros(u120.shape[0])
for i in range(u60.shape[0]):
    utau_60[i] = newton_raphson(zc_cord[id1], u60[i], utau_in_60[i], mu)[0]
    grad_model_u60[i] = rho  * (utau_60[i]**2)
    utau_120[i] = newton_raphson(zc_cord[id2], u120[i], utau_in_120[i], mu)[0]
    grad_model_u120[i] = rho * (utau_120[i]**2)

err_u60 = np.abs(grad_model_u60 - target)
err_u120 = np.abs(grad_model_u120 - target)
grad_model = np.where(err_u60 <= err_u120, grad_model_u60, grad_model_u120)
model_comparison = np.where(err_u60 <= err_u120, 60, 120)
count_60 = np.sum(model_comparison == 60)
count_120 = np.sum(model_comparison == 120)
print(model_comparison.shape)
print('model y+ = 60 better in {}'.format(count_60))
print('model y+ = 120 better in {}'.format(count_120))

risultati_EWM = np.zeros((grad_model.shape[0],6))
risultati_EWM[:,0] = target
risultati_EWM[:,1] = grad_model_u60
risultati_EWM[:,2] = grad_model_u120
risultati_EWM[:,3] = grad_model
risultati_EWM[:,4] = err_u60
risultati_EWM[:,5] = err_u120

np.save('risultati_EWM.npy',risultati_EWM)
model_comp_dict = {}
for i in range(0, model_comparison.shape[0], 7744):
    for j in range(1,68):
        nome = 'model_comp_{}'.format(j)
        model_comp_temp = model_comparison[i:i+7744]
        model_comp_temp = model_comp_temp.reshape(121,64)
        model_comp_dict[nome] = model_comp_temp
x_less = x[::10]
y_less = y[::10]
fig,ax = plt.subplots(3,2,figsize=(12,20))
ax[0,0].axis('scaled')
ax[0,1].axis('scaled')
ax[1,0].axis('scaled')
ax[1,1].axis('scaled')
ax[2,0].axis('scaled')
ax[2,1].axis('scaled')
p1 = ax[0,0].pcolormesh(x_less[:-1],y_less,model_comp_dict['model_comp_10'].T, cmap='viridis', shading='auto')
p2 = ax[0,1].pcolormesh(x_less[:-1],y_less,model_comp_dict['model_comp_20'].T, cmap='viridis', shading='auto')
p3 = ax[1,0].pcolormesh(x_less[:-1],y_less,model_comp_dict['model_comp_30'].T, cmap='viridis', shading='auto')
p4 = ax[1,1].pcolormesh(x_less[:-1],y_less,model_comp_dict['model_comp_40'].T, cmap='viridis', shading='auto')
p5 = ax[2,0].pcolormesh(x_less[:-1],y_less,model_comp_dict['model_comp_50'].T, cmap='viridis', shading='auto')
p6 = ax[2,1].pcolormesh(x_less[:-1],y_less,model_comp_dict['model_comp_60'].T, cmap='viridis', shading='auto')
ax[0,0].set_xlim(0,5)
ax[0,0].set_ylim(0,1.5)
ax[0,1].set_xlim(0,5)
ax[0,1].set_ylim(0,1.5)
ax[1,0].set_xlim(0,5)
ax[1,0].set_ylim(0,1.5)
ax[1,1].set_xlim(0,5)
ax[1,1].set_ylim(0,1.5)
ax[2,0].set_xlim(0,5)
ax[2,0].set_ylim(0,1.5)
ax[2,1].set_xlim(0,5)
ax[2,1].set_ylim(0,1.5)
plt.colorbar(p1, ax=ax[0,0])
plt.colorbar(p1, ax=ax[0,1])
plt.colorbar(p1, ax=ax[1,0])
plt.colorbar(p1, ax=ax[1,1])
plt.colorbar(p1, ax=ax[2,0])
plt.colorbar(p1, ax=ax[2,1])
plt.tight_layout()
plt.show()
plt.close()


#grad_model = np.zeros(u60.shape[0])
#for i in range(u60.shape[0]):
#    grad_60 = np.abs(grad_model_u60[i] - target[i])
#    grad_120 = np.abs(grad_model_u120[i] - target[i])
#    if (grad_60 <= grad_120):
#        grad_model[i] = grad_model_u60[i]
#    else:
#        grad_model[i] = grad_model_u120[i]
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

MSE_u60 = mean_squared_error(target / grad_wall_mean, grad_model_u60 / grad_wall_mean)
R2_u60 = r2_score(target / grad_wall_mean, grad_model_u60 / grad_wall_mean)
RMS_u60 = np.sqrt(MSE_u60)
MAE_u60 = mean_absolute_error(target/grad_wall_mean, grad_model_u60/grad_wall_mean)
print('prestazioni EWM ad u60')
print('MSE : {}'.format(MSE_u60))
print('MAE_EWM_60  = {}'.format(mean_absolute_error(target/grad_wall_mean, grad_model_u60/grad_wall_mean)))
print('R2 : {}'.format(R2_u60))
print('RMS : {}'.format(RMS_u60))

#sisultati modello u60
fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.scatterplot(x= target / grad_wall_mean , y= grad_model_u60 / grad_wall_mean, s=4, color='blue', ax=ax[0])
sns.kdeplot(x= target / grad_wall_mean, y= grad_model_u60 / grad_wall_mean, cmap='viridis', fill=True,  levels=30, thresh=0.01, bw_adjust=1, alpha=0.7, legend=True, ax=ax[1])
norm = plt.Normalize(vmin=0, vmax=5)  # valori arbitrari: personalizza!
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax[1])
cbar.set_label("Densità stimata")
plt.tight_layout()
ax[1].plot(target / grad_wall_mean, target / grad_wall_mean, color='red', linewidth=1)
ax[1].text(
    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
    "MSE: {:.4f}\nR2 : {:.4f}\nRMS : {:.4f}".format(MSE_u60, R2_u60, RMS_u60),     # testo da inserire
    transform=ax[1].transAxes,  # usa assi normalizzati (da 0 a 1)
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)
ax[0].plot(target / grad_wall_mean, target / grad_wall_mean, color='red', linewidth=1)
ax[0].set_xlabel('target / tauwall mean')
ax[0].set_ylabel('prediction / tauwall mean')
ax[0].set_title('Risultati EWM da yplus 60')
ax[1].set_xlabel('target / tauwall mean')
ax[1].set_ylabel('prediction / tauwall mean')
ax[1].set_title('Risultati EWM da yplus 60')
plt.show()
plt.savefig('EWM_con_filtro_e_subsample_u60.png', dpi=300)
plt.close()

fig,ax = plt.subplots(2,1,figsize=(4,5))
sns.histplot(data=grad_model_u60 / grad_wall_mean, stat='percent', color='r',bins=50, kde=True, ax = ax[0])
sns.histplot(data= target / grad_wall_mean, stat='percent', bins=50, kde=True, ax = ax[1])
ax[0].set_title('prevision')
ax[1].set_title('real value')
ax[0].set_xlim(target_normalized.min(), target_normalized.max())
ax[1].set_xlim(target_normalized.min(), target_normalized.max())
plt.tight_layout()
plt.show()
plt.savefig('distribuzione_risultati_u60.png', dpi=300)
plt.close()

MSE_u120 = mean_squared_error(target / grad_wall_mean, grad_model_u120 / grad_wall_mean)
R2_u120 = r2_score(target / grad_wall_mean, grad_model_u120 / grad_wall_mean)
RMS_u120 = np.sqrt(MSE_u120)
MAE_u120 = mean_absolute_error(target/grad_wall_mean, grad_model_u120/grad_wall_mean)
print('prestazioni EWM da yplus 120')
print('MSE : {}'.format(MSE_u120))
print('MAE_EWM_120  = {}'.format(mean_absolute_error(target/grad_wall_mean, grad_model_u120/grad_wall_mean)))
print('R2 : {}'.format(R2_u120))
print('RMS : {}'.format(RMS_u120))

#visualizza plot risultati
fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.scatterplot(x= target / grad_wall_mean , y= grad_model_u120 / grad_wall_mean, s=4, color='blue', ax=ax[0])
sns.kdeplot(x= target / grad_wall_mean, y= grad_model_u120 / grad_wall_mean, cmap='viridis', fill=True,  levels=30, thresh=0.01, bw_adjust=1, alpha=0.7, legend=True, ax=ax[1])
norm = plt.Normalize(vmin=0, vmax=5)  # valori arbitrari: personalizza!
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax[1])
cbar.set_label("Densità stimata")
plt.tight_layout()
ax[1].plot(target / grad_wall_mean, target / grad_wall_mean, color='red', linewidth=1)
ax[1].text(
    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
    "MSE: {:.4f}\nR2 : {:.4f}\nRMS : {:.4f}".format(MSE_u120, R2_u120, RMS_u120),     # te
    transform=ax[1].transAxes,  # usa assi normalizzati (da 0 a 1)
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)
ax[0].plot(target / grad_wall_mean, target / grad_wall_mean, color='red', linewidth=1)
ax[0].set_xlabel('target / tauwall mean')
ax[0].set_ylabel('prediction / tauwall mean')
ax[0].set_title('Risultati EWM da yplus 120')
ax[1].set_xlabel('target / tauwall mean')
ax[1].set_ylabel('prediction / tauwall mean')
ax[1].set_title('Risultati EWM da yplus 120')
plt.show()
plt.savefig('EWM_con_filtro_e_subsample_u120.png', dpi=300)
plt.close()

fig,ax = plt.subplots(2,1,figsize=(4,5))
sns.histplot(data=grad_model_u120 / grad_wall_mean, stat='percent', color='r',bins=50, kde=True, ax = ax[0])
sns.histplot(data= target / grad_wall_mean, stat='percent', bins=50, kde=True, ax = ax[1])
ax[0].set_xlim(target_normalized.min(), target_normalized.max())
ax[1].set_xlim(target_normalized.min(), target_normalized.max())
ax[0].set_title('prevision')
ax[1].set_title('real value')
plt.tight_layout()
plt.show()
plt.savefig('distribuzione_risultati_u120.png', dpi=300)
plt.close()

MSE_F = mean_squared_error(target / grad_wall_mean, grad_model / grad_wall_mean)
R2_F = r2_score(target / grad_wall_mean, grad_model / grad_wall_mean)
RMS_F = np.sqrt(MSE_F)
MAE_F = mean_absolute_error(target/grad_wall_mean, grad_model/grad_wall_mean)
print('prestazioni modello EWM best of two')
print('MSE : {}'.format(MSE_F))
print('MAE_EWM_best  = {}'.format(mean_absolute_error(target/grad_wall_mean, grad_model/grad_wall_mean)))
print('R2 : {}'.format(R2_F))
print('RMS : {}'.format(RMS_F))

#risultati EWM con grad_model come trade-off tra i due modelli
fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.scatterplot(x= target / grad_wall_mean , y= grad_model / grad_wall_mean, s=4, color='blue', ax=ax[0])
sns.kdeplot(x= target / grad_wall_mean, y= grad_model / grad_wall_mean, cmap='viridis', fill=True,  levels=30, thresh=0.01, bw_adjust=1, alpha=0.7, cbar=True, legend=True, ax=ax[1])
ax[1].plot(target / grad_wall_mean, target / grad_wall_mean, color='red', linewidth=1)
ax[1].text(
    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
    "MSE: {:.4f}\nR2 : {:.4f}\nRMS : {:.4f}".format(MSE_F, R2_F, RMS_F),     # te
    transform=ax[1].transAxes,  # usa assi normalizzati (da 0 a 1)
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)
ax[0].plot(target / grad_wall_mean, target / grad_wall_mean, color='red', linewidth=1)
ax[0].set_xlabel('target / tauwall mean')
ax[0].set_ylabel('prediction / tauwall mean')
ax[0].set_title('Risultati EWM')
ax[1].set_xlabel('target / tauwall mean')
ax[1].set_ylabel('prediction / tauwall mean')
ax[1].set_title('Risultati EWM')
plt.show()
plt.savefig('EWM_con_filtro_e_subsample.png', dpi=300)
plt.close()

fig,ax = plt.subplots(2,1,figsize=(4,5))
sns.histplot(data=grad_model / grad_wall_mean, stat='percent', color='r',bins=50, kde=True, ax = ax[0])
sns.histplot(data= target / grad_wall_mean, stat='percent', bins=50, kde=True, ax = ax[1])
ax[0].set_xlim(target_normalized.min(), target_normalized.max())
ax[1].set_xlim(target_normalized.min(), target_normalized.max())
ax[0].set_title('prevision')
ax[1].set_title('real value')
plt.tight_layout()
plt.show()
plt.savefig('distribuzione_risultati.png', dpi=300)
plt.close()

# Confronto grafico delle performance
models = ["EWM_u60", "EWM_u120", "EWM_blended"]
mse_scores = [MSE_u60, MSE_u120,MSE_F]
mae_scores = [MAE_u60, MAE_u120,MAE_F]

r2_scores = [R2_u60, R2_u120, R2_F]
rms_scores = [RMS_u60, RMS_u120, RMS_F]

plt.figure(figsize=(18,5))

plt.subplot(1,4,1)
plt.bar(models, mse_scores, color=['blue', 'green','red'])
plt.ylabel("MSE (Errore Minore è Meglio)")
plt.title("Confronto MSE")

plt.subplot(1,4,1)
plt.bar(models, mae_scores, color=['blue', 'green','red'])
plt.ylabel("MSE (Errore Minore è Meglio)")
plt.title("Confronto MSE")

plt.subplot(1,4,3)
plt.bar(models, r2_scores, color=['blue', 'green', 'red'])
plt.ylabel("R² Score (Più Alto è Meglio)")
plt.title("Confronto R² Score")

plt.subplot(1,4,4)
plt.bar(models, rms_scores, color=['blue', 'green', 'red'])
plt.ylabel("RMS Score (Minore è Meglio)")
plt.title("Confronto RMS Score")

plt.savefig('confronto_EWM.png', dpi=300)
plt.show()
plt.close()

#dataset = np.load('cfd_ml_update/DATA/dataset_retau_1000_subsamplemean_senzagauss_nondimensional.npy')
dataset = np.zeros((grad_model_u60.shape[0],7))
utau60 = np.sqrt( (mu / rho) * grad_model_u60)
utau120 = np.sqrt( (mu / rho) * grad_model_u120)
utau_target = np.sqrt( (mu / rho) * target)
dataset[:,0] = target
dataset[:,1] = u60 / utau60 
dataset[:,2] = v60 / utau60
dataset[:,3] = w60 / utau60
dataset[:,4] = u120 / utau120 
dataset[:,5] = v120 / utau120
dataset[:,6] = w120 / utau120
np.save('cfd_ml_update/DATA/EWM_dataset_submeanonly_adimensionalized_gradastarget.npy', dataset)

#dataset[:,4] = grad_model 
#np.save('cfd_ml_update/DATA/EWM_dataset_sub_mean_grad_model_blended.npy', dataset)
#
#dataset_reduce = np.zeros((int(dataset.shape[0]),6))
#dataset_reduce[:,0] = dataset[:,0]
#dataset_reduce[:,1] = grad_model
#dataset_reduce[:,2] = dataset[:,2]
#dataset_reduce[:,3] = dataset[:,3]
#dataset_reduce[:,4] = dataset[:,5]
#dataset_reduce[:,5] = dataset[:,6]
#
#np.save('cfd_ml_update/DATA/EWM_dataset_sub_mean_grad_model_only.npy', dataset_reduce)


