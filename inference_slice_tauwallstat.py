import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew,kurtosis
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch.nn.init as init

from accelerator_conditional import Cfd_mlp
import joblib
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

y_train = np.load('dataset_dict/tauwall_scaled_train_ravel.npy')
hist, bin_edges = np.histogram(y_train, bins=100, density=True)
bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2
from scipy.interpolate import interp1d
freq_interp = interp1d(bin_center, hist, bounds_error=False, fill_value='extrapolate')

model = Cfd_mlp.load_from_checkpoint("/leonardo_work/IscrC_MLWM-CF/last_EWM/checkpoints_512_256_128_ReduceOnPlateau_MSE/epoch=27-val_loss=0.8272191_.ckpt",
                                     batch_size = 128,
                                     learning_rate = 1e-4,
                                     node_per_layer = [9, 512, 256, 128],
                                     freq_interp = freq_interp
                                     )
model.eval()
model.to("cuda")

tauwall_test = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/dataset_dict/tauwall_scaled_test.npz'))
u60_test = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/dataset_dict/u60_scaled_test.npz'))
v60_test = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/dataset_dict/v60_scaled_test.npz'))
w60_test = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/dataset_dict/w60_scaled_test.npz'))
u120_test = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/dataset_dict/u120_scaled_test.npz'))
v120_test = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/dataset_dict/v120_scaled_test.npz'))
w120_test = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/dataset_dict/w120_scaled_test.npz'))
delta_u_test = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/dataset_dict/delta_u_scaled_test.npz'))
delta_v_test = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/dataset_dict/delta_v_scaled_test.npz'))
delta_w_test = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/dataset_dict/delta_w_scaled_test.npz'))
ss_u60  = joblib.load('scaler/u60_scaler.pkl') 
ss_v60  = joblib.load('scaler/v60_scaler.pkl') 
ss_w60  = joblib.load('scaler/w60_scaler.pkl') 
ss_u120 = joblib.load('scaler/u120_scaler.pkl') 
ss_v120 = joblib.load('scaler/v120_scaler.pkl') 
ss_w120 = joblib.load('scaler/w120_scaler.pkl') 
ss_tauwall = joblib.load('scaler/tau_wall_scaler.pkl')
ss_delta_u = joblib.load('scaler/deltau_scaler.pkl')
ss_delta_v = joblib.load('scaler/deltav_scaler.pkl')
ss_delta_w = joblib.load('scaler/deltaw_scaler.pkl')

#pt_target = joblib.load('pt_target_splitted.pkl')

Retau_mean = 1.02003270E+003
utau_mean  = 4.97576926E-002
deltav = 0.5 / Retau_mean
mu = deltav * utau_mean
y_wall = 0.0001442222
y60 = 0.02945271
y120 = 0.05834579
#U_bulk_plus = 21.1 #valore tipico per canali lisci a retau1000 mi serve solo per adimensionalizzare
#Re_bulk = U_bulk_plus * 2 *Retau
Re_bulk = 41000
u_bulk = Re_bulk * mu / (2*0.5)


prediction_test = {}
prediction_test_norm = {}
for (key, chiaveu60, chiavev60, chiavew60, chiaveu120, chiavev120,chiavew120) in zip(tauwall_test.keys(), u60_test.keys(), v60_test.keys(), w60_test.keys(), u120_test.keys(), v120_test.keys(), w120_test.keys()):

    u60 = u60_test[chiaveu60].ravel()
    v60 = v60_test[chiavev60].ravel()
    w60 = w60_test[chiavew60].ravel()
    u120 = u120_test[chiaveu120].ravel()
    v120 = v120_test[chiavev120].ravel()
    w120 = w120_test[chiavew120].ravel()
    delta_u = delta_u_test[chiaveu120].ravel()
    delta_v = delta_v_test[chiavev120].ravel()
    delta_w = delta_w_test[chiavew120].ravel()

    X_test = np.zeros((u60.shape[0], 9))
    X_test[:,0] = u60
    X_test[:,1] = w60
    X_test[:,2] = v60
    X_test[:,3] = u120
    X_test[:,4] = w120
    X_test[:,5] = u120
    X_test[:,6] = delta_u
    X_test[:,7] = delta_v
    X_test[:,8] = delta_w


    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad(): 
        pred_norm = model(X_test)

       # pred = pred.detach().cpu().numpy() 
       # pred = np.exp(pred) -1
        pred_norm = pred_norm.detach().cpu().numpy()
        pred = ss_tauwall.inverse_transform(pred_norm.reshape(-1,1))
        pred = pred * (u_bulk**2)
        pred_norm = pred_norm.reshape(121,64)
        prediction_test_norm[key] = pred_norm
        pred = pred.reshape(121,64)
        prediction_test[key] = pred

#normalizzo il target
tauwall_test_dim= {}
for key in tauwall_test.keys():
    campo = tauwall_test[key].ravel()
    campo = ss_tauwall.inverse_transform(campo.reshape(-1,1)).reshape(121,64)
    campo = campo * (u_bulk**2)
    tauwall_test_dim[key] = campo

Retau_mean = 1.02003270E+003
utau_mean  = 4.97576926E-002
lz = 1
delta  = lz / 2
delta_v = delta / Retau_mean
rho = 1.
mu = utau_mean * delta_v
tauwall_stat = rho * (utau_mean**2)


lx = 5.
ly = 1.5
nx = 1216
ny = 640

x = np.linspace(0,lx,nx)
y = np.linspace(0,ly,ny)

x = x[:(x.shape[0] //10)*10]
y = y[:(y.shape[0] //10)*10]
x_less = x[::10]
y_less = y[::10]

pred_errore = {}
for (key,keys) in zip(tauwall_test_dim.keys(),prediction_test.keys()):
    pred_errore[key] = (prediction_test[keys] - tauwall_test_dim[key]) / tauwall_test_dim[key]
    fig,ax = plt.subplots(3,1,figsize=(12,20))#, constrained_layout=True)
    utaumean_pred = np.mean(prediction_test[keys], axis=(0,1))
    utaumean_test = np.mean(tauwall_test_dim[key], axis=(0,1))
    fig.suptitle('tauwall_test_mean = {:.5f}, tauwall_pred_mean = {:.5f}'.format(utaumean_test, utaumean_pred))
    ax[0].axis('scaled')
    ax[1].axis('scaled')
    ax[2].axis('scaled')
    vmin = 0.0004#min(tauwall_test[key].min(), prediction_test[keys].min())
    vmax = 0.006#max(tauwall_test[key].max(), prediction_test[keys].max())
    vmin_errore = -1
    vmax_errore = 1
    s1 = ax[0].pcolormesh(x_less,y_less,tauwall_test_dim[key].T, vmin=vmin, vmax=vmax, cmap='coolwarm', shading='auto')
    s2 = ax[1].pcolormesh(x_less,y_less,prediction_test[keys].T, vmin=vmin, vmax=vmax, cmap='coolwarm', shading='auto')
    e1 = ax[2].pcolormesh(x_less,y_less,pred_errore[key].T, vmin=vmin_errore, vmax=vmax_errore, cmap='coolwarm', shading='auto')
    ax[0].set_xlim(0,5)
    ax[0].set_ylim(0,1.5)
    ax[1].set_xlim(0,5)
    ax[1].set_ylim(0,1.5)
    ax[2].set_xlim(0,5)
    ax[2].set_ylim(0,1.5)
    #fig.colorbar(s1, ax = [ax[0], ax[1], ax[2]], orientation='vertical', shrink=0.8, pad=1)
    #plt.tight_layout()
    ax[0].set_title('tauwall reference {}'.format(key))
    ax[1].set_title('tauwall predict {}'.format(keys))
    ax[2].set_title('relative_error {}'.format(key))
    # Posizione manuale della colorbar (x0, y0, width, height)
    cbar_ax = fig.add_axes([0.91, 0.37, 0.01, 0.5])
    cbar_ax1 = fig.add_axes([0.91, 0.03, 0.01, 0.3])
    fig.colorbar(s1, cax=cbar_ax, label='Valore')
    fig.colorbar(e1, cax=cbar_ax1, label='Errore')
    plt.subplots_adjust(right=0.9)  # lascia spazio alla colorbar
    plt.savefig('Slice_MLP_MSE_ReduceONPlateau/Slice_{}.png'.format(key), dpi=300)
    #plt.show()
    plt.close()

y_hat = np.array([])
y = np.array([])
for key in prediction_test.keys():
    campo_pred = prediction_test[key].ravel()
    campo_target = tauwall_test_dim[key].ravel()
    y_hat = np.append(y_hat, campo_pred)
    y = np.append(y, campo_target)

mse = mean_squared_error(y , y_hat )
r2 = r2_score(y , y_hat )
rms = np.sqrt(mse)
print('prestazioni modello Retau1000 MSE ReduceOnPlateau, Dropout=0.05')
print('MSE = {}'.format(mse))
print('MAE = {}'.format(mean_absolute_error(y , y_hat)))
print('R2 = {}'.format(r2))
print('RMS = {}'.format(rms))

errore = np.abs(y - y_hat)
print('errore minimo = {}'.format(errore.min()))
print('errore massimo = {}'.format(errore.max()))
print('85esimo percentile test_set = {}'.format(np.percentile(errore, 85)))

fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.scatterplot(x= y , y= y_hat, s=4, color='blue', ax=ax[0])
sns.kdeplot(x= y, y= y_hat, cmap='coolwarm', fill=True,  levels=30, thresh=0.01, bw_adjust=1, alpha=0.7, legend=True, ax=ax[1])
norm = plt.Normalize(vmin=0, vmax=5)  # valori arbitrari: personalizza!
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax[1])
cbar.set_label("Density")
plt.tight_layout()
ax[1].plot(y, y, color='black', linewidth=1)
ax[1].text(
    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
    "MSE: {:.5f}\nR2 : {:.5f}\nRMS : {:.5f}".format(mse, r2, rms),     # te
    transform=ax[1].transAxes,  # usa assi normalizzati (da 0 a 1)
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)
ax[0].plot(y, y, color='black', linewidth=1)
ax[0].set_xlabel('tauwall DNS')
ax[0].set_ylabel('tauwall NN')
ax[0].set_xlim(0,0.007)
ax[0].set_ylim(0,0.007)
ax[1].set_xlim(0,0.007)
ax[1].set_ylim(0,0.007)
ax[0].set_title('Risultati NN yplus 60-120')
ax[1].set_xlabel('tauwall DNS')
ax[1].set_ylabel('tauwall NN')
ax[1].set_title('Risultati NN  yplus 60-120')
plt.show()
plt.savefig('Slice_MLP_MSE_ReduceONPlateau/NN_DNS_filtered.png', dpi=300)
plt.close()

fig,ax = plt.subplots()
sns.histplot(y, stat='density', bins=100, color='blue', kde=True, ax=ax)
ax.axvline(np.mean(y), color='black')
ax.text(
    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
    "MEAN: {:.4f}\nSTD: {:.4f}".format(np.mean(y), np.std(y)),     # te
    transform=ax.transAxes,  # usa assi normalizzati (da 0 a 1)
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)

ax.set_title('target')
ax.set_yscale('log')
ax.set_ylim(1,1400)
ax.set_xlim(0.000, 0.007)
plt.savefig('Slice_MLP_MSE_ReduceONPlateau/target_distribution_log.png', dpi=300)

fig,ax = plt.subplots()
sns.histplot(y_hat, stat='density', bins=100, color='red', kde=True, ax=ax)
ax.axvline(np.mean(y_hat), color='black')
ax.text(
    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
    "MEAN: {:.4f}\nSTD: {:.4f}".format(np.mean(y_hat), np.std(y_hat)),     # te
    transform=ax.transAxes,  # usa assi normalizzati (da 0 a 1)
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)
ax.set_title('NN prediction')
ax.set_yscale('log')
ax.set_ylim(1,1480)
ax.set_xlim(0.000, 0.007)
plt.savefig('Slice_MLP_MSE_ReduceONPlateau/NN_distribution_log.png', dpi=300)
exit()





#y_hat_norm = np.array([])
#y_norm = np.array([])
#for key in prediction_test_norm.keys():
#    campo_pred = prediction_test_norm[key].ravel()
#    campo_target = tauwall_test_norm[key].ravel()
#    y_hat_norm = np.append(y_hat_norm, campo_pred)
#    y_norm = np.append(y_norm, campo_target)
#
#mse = mean_squared_error(y_norm , y_hat_norm )
#r2 = r2_score(y_norm , y_hat_norm )
#rms = np.sqrt(mse)
#print('prestazioni modello Retau1000 splitted LeakyReLU, huber_loss, Dropout=0.05 NORM')
#print('MSE : {}'.format(mse))
#print('MAE : {}'.format(mean_absolute_error(y_norm , y_hat_norm)))
#print('R2 : {}'.format(r2))
#print('RMS : {}'.format(rms))
#
#errore_norm = np.abs(y_norm - y_hat_norm)
#print('errore minimo = {}'.format(errore_norm.min()))
#print('errore massimo = {}'.format(errore_norm.max()))
#print('85esimo percentile test_set = {}'.format(np.percentile(errore_norm, 85)))
#
#tauwall_train = dict(np.load('/home/dalmonte/CFD/MLWM/tauwall_train_utau_tauwall_stat.npz'))
#u60_train = dict(np.load('/home/dalmonte/CFD/MLWM/u60_train_utau_tauwall_stat.npz'))
#v60_train = dict(np.load('/home/dalmonte/CFD/MLWM/v60_train_utau_tauwall_stat.npz'))
#w60_train = dict(np.load('/home/dalmonte/CFD/MLWM/w60_train_utau_tauwall_stat.npz'))
#u120_train = dict(np.load('/home/dalmonte/CFD/MLWM/u120_train_utau_tauwall_stat.npz'))
#v120_train = dict(np.load('/home/dalmonte/CFD/MLWM/v120_train_utau_tauwall_stat.npz'))
#w120_train = dict(np.load('/home/dalmonte/CFD/MLWM/w120_train_utau_tauwall_stat.npz'))
#
#prediction_train = {}
#prediction_train_norm = {}
#for (key, chiaveu60, chiavev60, chiavew60, chiaveu120, chiavev120,chiavew120) in zip(tauwall_train.keys(), u60_train.keys(), v60_train.keys(), w60_train.keys(), u120_train.keys(), v120_train.keys(), w120_train.keys()):
#
#    u60 = u60_train[chiaveu60].ravel()
#    v60 = v60_train[chiavev60].ravel()
#    w60 = w60_train[chiavew60].ravel()
#    u120 = u120_train[chiaveu120].ravel()
#    v120 = v120_train[chiavev120].ravel()
#    w120 = w120_train[chiavew120].ravel()
#    u60 = ss_u60.transform(u60.reshape(-1,1)).ravel()
#    v60 = ss_v60.transform(v60.reshape(-1,1)).ravel()
#    w60 = ss_w60.transform(w60.reshape(-1,1)).ravel()
#    u120 = ss_u120.transform(u120.reshape(-1,1)).ravel()
#    v120 = ss_v120.transform(v120.reshape(-1,1)).ravel()
#    w120 = ss_w120.transform(w120.reshape(-1,1)).ravel()
#   # u60 = mm_u60.transform(u60.reshape(-1,1)).ravel()
#   # v60 = mm_v60.transform(v60.reshape(-1,1)).ravel()
#   # w60 = mm_w60.transform(w60.reshape(-1,1)).ravel()
#   # u120 = mm_u120.transform(u120.reshape(-1,1)).ravel()
#   # v120 = mm_v120.transform(v120.reshape(-1,1)).ravel()
#   # w120 = mm_w120.transform(w120.reshape(-1,1)).ravel()
#
#    X_train = np.zeros((u60.shape[0], 6))
#    X_train[:,0] = u60
#    X_train[:,1] = v60
#    X_train[:,2] = w60
#    X_train[:,3] = u120
#    X_train[:,4] = v120
#    X_train[:,5] = w120
#
#    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
#    with torch.no_grad():
#        pred_norm = model(X_train)
#
#       # pred = pred.detach().cpu().numpy() 
#       # pred = np.exp(pred) -1
#        pred_norm = pred_norm.detach().cpu().numpy()
#        pred = pt_target.inverse_transform(pred_norm.reshape(-1,1))
#        pred_norm = pred_norm.reshape(121,64)
#        prediction_train_norm[key] = pred_norm
#        pred = pred.reshape(121,64)
#        prediction_train[key] = pred
##np.savez('/home/dalmonte/CFD/MLWM/tauwall_test_NN_dimensionalized_tauwallstat.npz', **tauwall_test)
##np.savez('/home/dalmonte/CFD/MLWM/prediction_LR_HL_tauwallstat.npz', **prediction_test)
##np.savez('/home/dalmonte/CFD/MLWM/errore_LR_HL_tauwallstat.npz', **pred_errore)
#
##normalizzo il target
#tauwall_train_norm = {}
#for key in tauwall_train.keys():
#    campo = tauwall_train[key]
#    campo = campo.ravel()
#    campo = pt_target.transform(campo.reshape(-1,1))
#    tauwall_train_norm[key] = campo
#
#for key in tauwall_train.keys():
#        tauwall_train[key] = tauwall_train[key] * tauwall_stat
#        prediction_train[key] = prediction_train[key] * tauwall_stat
#
#pred_errore_train = {}
#for (key,keys) in zip(tauwall_train.keys(),prediction_train.keys()):
#    pred_errore_train[key] = (prediction_train[keys] - tauwall_train[key]) / tauwall_stat
#    fig,ax = plt.subplots(3,1,figsize=(12,20))#, constrained_layout=True)
#    utaumean_pred = np.mean(prediction_train[key], axis=(0,1))
#    utaumean_train = np.mean(tauwall_train[key], axis=(0,1))
#    fig.suptitle('tauwall_train_mean = {:.5f}, tauwall_pred_mean = {:.5f}'.format(utaumean_train, utaumean_pred))
#    ax[0].axis('scaled')
#    ax[1].axis('scaled')
#    ax[2].axis('scaled')
#    vmin = 0.0004#min(tauwall_test[key].min(), prediction_test[keys].min())
#    vmax = 0.006#max(tauwall_test[key].max(), prediction_test[keys].max())
#    vmin_errore = -1
#    vmax_errore = 1
#    s1 = ax[0].pcolormesh(x_less[:-1],y_less,tauwall_train[key].T, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
#    s2 = ax[1].pcolormesh(x_less[:-1],y_less,prediction_train[keys].T, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
#    e1 = ax[2].pcolormesh(x_less[:-1],y_less,pred_errore_train[key].T, vmin=vmin_errore, vmax=vmax_errore, cmap='coolwarm', shading='auto')
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
#    ax[2].set_title('errore {}'.format(key))
#    # Posizione manuale della colorbar (x0, y0, width, height)
#    cbar_ax = fig.add_axes([0.91, 0.37, 0.01, 0.5])
#    cbar_ax1 = fig.add_axes([0.91, 0.03, 0.01, 0.3])
#    fig.colorbar(s1, cax=cbar_ax, label='Valore')
#    fig.colorbar(e1, cax=cbar_ax1, label='Errore')
#    plt.subplots_adjust(right=0.9)  # lascia spazio alla colorbar
#    plt.savefig('Slice_leakyReLU_huberloss_tauwallstat/TRAIN_tauwallstat_morelayer_{}.png'.format(key), dpi=300)
#    #plt.show()
#    plt.close()
#
#mse = mean_squared_error(y , y_hat )
#r2 = r2_score(y , y_hat )
#rms = np.sqrt(mse)
#print('prestazioni modello Retau1000 splitted LeakyReLU, huber_loss, Dropout=0.05')
#print('MSE : {}'.format(mse))
#print('MAE : {}'.format(mean_absolute_error(y , y_hat)))
#print('R2 : {}'.format(r2))
#print('RMS : {}'.format(rms))
#
#errore = np.abs(y - y_hat)
#print('errore minimo = {}'.format(errore.min()))
#print('errore massimo = {}'.format(errore.max()))
#print('85esimo percentile test_set = {}'.format(np.percentile(errore, 85)))
#
#y_hat_norm = np.array([])
#y_norm = np.array([])
#for key in prediction_test_norm.keys():
#    campo_pred = prediction_test_norm[key].ravel()
#    campo_target = tauwall_test_norm[key].ravel()
#    y_hat_norm = np.append(y_hat_norm, campo_pred)
#    y_norm = np.append(y_norm, campo_target)
#
#mse = mean_squared_error(y_norm , y_hat_norm )
#r2 = r2_score(y_norm , y_hat_norm )
#rms = np.sqrt(mse)
#print('prestazioni modello Retau1000 splitted LeakyReLU, huber_loss, Dropout=0.05 NORM')
#print('MSE : {}'.format(mse))
#print('MAE : {}'.format(mean_absolute_error(y_norm , y_hat_norm)))
#print('R2 : {}'.format(r2))
#print('RMS : {}'.format(rms))
#
#errore_norm = np.abs(y_norm - y_hat_norm)
#print('errore minimo = {}'.format(errore_norm.min()))
#print('errore massimo = {}'.format(errore_norm.max()))
#print('85esimo percentile test_set = {}'.format(np.percentile(errore_norm, 85)))    
