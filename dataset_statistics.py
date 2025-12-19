import numpy as np
from scipy.stats import skew, kurtosis

u60 = dict(np.load('slice_u60_less_train.npz'))
v60 = dict(np.load('slice_v60_less_train.npz'))
w60 = dict(np.load('slice_w60_less_train.npz'))   #è y
u120 = dict(np.load('slice_u120_less_train.npz'))
v120 = dict(np.load('slice_v120_less_train.npz'))
w120 = dict(np.load('slice_w120_less_train.npz'))
uwall = dict(np.load('slice_u_wall_less_train.npz'))
delta_u = dict(np.load('delta_u_less_train.npz'))
delta_v = dict(np.load('delta_v_less_train.npz'))
delta_w = dict(np.load('delta_w_less_train.npz'))

#u60   = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u60.npz'))
#v60   = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_v60.npz'))
#w60   = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_w60.npz'))   #è y
#u120  = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u120.npz'))
#v120  = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_v120.npz'))
#w120  = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_w120.npz'))
#uwall = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u_wall.npz'))
#delta_u = {}
#delta_v = {}
#delta_w = {}
#for (keyu60, keyv60, keyw60, keyu120, keyv120, keyw120) in zip(u60.keys(), v60.keys(), w60.keys(), u120.keys(), v120.keys(), w120.keys()):
#    campo_u = u120[keyu120] - u60[keyu60]
#    campo_v = v120[keyv120] - v60[keyv60]
#    campo_w = w120[keyw120] - w60[keyw60]
#    delta_u[keyu120] = campo_u
#    delta_v[keyv120] = campo_v
#    delta_w[keyw120] = campo_w
#np.savez('delta_u.npz',**delta_u)
#np.savez('delta_v.npz',**delta_v)
#np.savez('delta_w.npz',**delta_w)
#
#exit()

y60 = 0.02945271
y120 = 0.05834579
delta = 0.5
ni = 2.439024386178992e-05

u60_ravel = np.array([])
v60_ravel = np.array([])
w60_ravel = np.array([])
u120_ravel = np.array([])
v120_ravel = np.array([])
w120_ravel = np.array([])
uwall_ravel = np.array([])
delta_u_ravel = np.array([])
delta_v_ravel = np.array([])
delta_w_ravel = np.array([])

for (kwall, ku60, kv60, kw60, ku120,kv120,kw120) in zip(uwall.keys(), u60.keys(),v60.keys(), w60.keys(),u120.keys(),v120.keys(),w120.keys()):
    u60_ravel = np.append(u60_ravel, u60[ku60].ravel())
    v60_ravel = np.append(v60_ravel, v60[kv60].ravel())
    w60_ravel = np.append(w60_ravel, w60[kw60].ravel())
    u120_ravel = np.append(u120_ravel, u120[ku120].ravel())
    v120_ravel = np.append(v120_ravel, v120[kv120].ravel())
    w120_ravel = np.append(w120_ravel, w120[kw120].ravel())
    uwall_ravel = np.append(uwall_ravel, uwall[kwall].ravel())
    delta_u_ravel = np.append(delta_u_ravel, delta_u[ku120].ravel())
    delta_v_ravel = np.append(delta_v_ravel, delta_v[kv120].ravel())
    delta_w_ravel = np.append(delta_w_ravel, delta_w[kw120].ravel())
def stat(campo, nome):
    sk = skew(campo)
    kur = kurtosis(campo)
    print('{} --> skew = {:.5f}, kurtosis = {:.5f}'.format(nome, sk, kur))

campo = ['uwall', 'u60', 'v60','w60','u120','v120','w120','deltau','deltav','deltaw']
stat(uwall_ravel, campo[0])
stat(u60_ravel, campo[1])
stat(v60_ravel, campo[2])
stat(w60_ravel, campo[3])
stat(u120_ravel, campo[4])
stat(v120_ravel, campo[5])
stat(w120_ravel, campo[6])
stat(delta_u_ravel, campo[7])
stat(delta_v_ravel, campo[8])
stat(delta_w_ravel, campo[9])

print('|skew| > 0.3 e/o kurtosis > 1.0 suggeriscono che potresti aver bisogno di trasformazioni.')
#y60 = y60 / delta
#y120 = y120 / delta3

