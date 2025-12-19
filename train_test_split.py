import numpy as np

#u60 = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u60.npz'))
#v60 = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_v60.npz'))
#w60 = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_w60.npz'))
#u120 = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u120.npz'))
#v120 = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_v120.npz'))
#w120 = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_w120.npz'))
#uwall = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u_wall.npz'))

delta_u = dict(np.load('delta_u.npz'))
delta_v = dict(np.load('delta_v.npz'))
delta_w = dict(np.load('delta_w.npz'))

train_index = [59, 53, 45, 52, 13, 57, 38, 65, 2, 14, 56, 42, 23, 11, 48, 51, 36, 4, 3, 24, 10, 12, 18, 8, 35, 30, 61, 17, 66, 1, 34, 5, 31, 40, 7, 27, 43, 41, 46, 64, 20, 33, 63, 32, 28, 62, 55]

index = list(range(1,68))
rest = set(index) - set(train_index)
rest = list(rest)

valid_index = [50, 25, 29, 58, 54, 26, 44]
test_index = set(rest) - set(valid_index)
test_index = list(test_index)

u60_train = {}
u120_train = {}
uwall_train = {}
delta_u_train = {}
for i in train_index:
    #nome60 = 'u_60_{}'.format(i)
    nome120 = 'u_120_{}'.format(i)
    #nomewall = 'u_wall_{}'.format(i)
    #u60_train[nome60] = u60[nome60]
    #u120_train[nome120] = u120[nome120]
    #uwall_train[nomewall] = uwall[nomewall]
    delta_u_train[nome120] = delta_u[nome120]

v60_train = {}
v120_train = {}
delta_v_train = {}
for i in train_index:
    #nome60 = 'v_60_{}'.format(i)
    nome120 = 'v_120_{}'.format(i)
    #v60_train[nome60] = v60[nome60]
    #v120_train[nome120] = v120[nome120]
    delta_v_train[nome120] = delta_v[nome120]

w60_train = {}
w120_train = {}
delta_w_train = {}
for i in train_index:
    #nome60 = 'w_60_mean_{}'.format(i)
    nome120 = 'w_120_mean_{}'.format(i)
    #w60_train[nome60] = w60[nome60]
    #w120_train[nome120] = w120[nome120]
    delta_w_train[nome120] = delta_w[nome120]


u60_test = {}
u120_test = {}
uwall_test = {}
delta_u_test = {}
for i in test_index:
    #nome60 = 'u_60_{}'.format(i)
    nome120 = 'u_120_{}'.format(i)
    #nomewall = 'u_wall_{}'.format(i)
    #u60_test[nome60] = u60[nome60]
    #u120_test[nome120] = u120[nome120]
    #uwall_test[nomewall] = uwall[nomewall]
    delta_u_test[nome120] = delta_u[nome120]

v60_test = {}
v120_test = {}
delta_v_test = {}
for i in test_index:
    #nome60 = 'v_60_{}'.format(i)
    nome120 = 'v_120_{}'.format(i)
    #v60_test[nome60] = v60[nome60]
    #v120_test[nome120] = v120[nome120]
    delta_v_test[nome120] = delta_v[nome120]

w60_test = {}
w120_test = {}
delta_w_test = {}
for i in test_index:
    #nome60 = 'w_60_mean_{}'.format(i)
    nome120 = 'w_120_mean_{}'.format(i)
    #w60_test[nome60] = w60[nome60]
    #w120_test[nome120] = w120[nome120]
    delta_w_test[nome120] = delta_w[nome120]

u60_valid = {}
u120_valid = {}
uwall_valid = {}
delta_u_valid = {}
for i in valid_index:
    #nome60 = 'u_60_{}'.format(i)
    nome120 = 'u_120_{}'.format(i)
    #nomewall = 'u_wall_{}'.format(i)
    #u60_valid[nome60] = u60[nome60]
    #u120_valid[nome120] = u120[nome120]
    #uwall_valid[nomewall] = uwall[nomewall]
    delta_u_valid[nome120] = delta_u[nome120]

v60_valid = {}
v120_valid = {}
delta_v_valid = {}
for i in valid_index:
    #nome60 = 'v_60_{}'.format(i)
    nome120 = 'v_120_{}'.format(i)
    #v60_valid[nome60] = v60[nome60]
    #v120_valid[nome120] = v120[nome120]
    delta_v_valid[nome120] = delta_v[nome120]

w60_valid = {}
w120_valid = {}
delta_w_valid = {}
for i in valid_index:
    #nome60 = 'w_60_mean_{}'.format(i)
    nome120 = 'w_120_mean_{}'.format(i)
    #w60_valid[nome60] = w60[nome60]
    #w120_valid[nome120] = w120[nome120]
    delta_w_valid[nome120] = delta_w[nome120]


np.savez('delta_u_train.npz', **delta_u_train)
np.savez('delta_u_test.npz',  **delta_u_test)
np.savez('delta_u_valid.npz', **delta_u_valid)
np.savez('delta_v_train.npz', **delta_v_train)
np.savez('delta_v_test.npz',  **delta_v_test)
np.savez('delta_v_valid.npz', **delta_v_valid)
np.savez('delta_w_train.npz', **delta_w_train)
np.savez('delta_w_test.npz',  **delta_w_test)
np.savez('delta_w_valid.npz', **delta_w_valid)

#np.savez('slice_u60_train.npz', **u60_train)
#np.savez('slice_u60_test.npz',  **u60_test)
#np.savez('slice_u60_valid.npz', **u60_valid)
#np.savez('slice_v60_train.npz', **v60_train)
#np.savez('slice_v60_test.npz',  **v60_test)
#np.savez('slice_v60_valid.npz', **v60_valid)
#np.savez('slice_w60_train.npz', **w60_train)
#np.savez('slice_w60_test.npz',  **w60_test)
#np.savez('slice_w60_valid.npz', **w60_valid)
#np.savez('slice_u120_train.npz', **u120_train)
#np.savez('slice_u120_test.npz',  **u120_test)
#np.savez('slice_u120_valid.npz', **u120_valid)
#np.savez('slice_v120_train.npz', **v120_train)
#np.savez('slice_v120_test.npz',  **v120_test)
#np.savez('slice_v120_valid.npz', **v120_valid)
#np.savez('slice_w120_train.npz', **w120_train)
#np.savez('slice_w120_test.npz',  **w120_test)
#np.savez('slice_w120_valid.npz', **w120_valid)
#np.savez('slice_u_wall_train.npz', **uwall_train)
#np.savez('slice_u_wall_test.npz',  **uwall_test)
#np.savez('slice_u_wall_valid.npz', **uwall_valid)
