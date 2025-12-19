import numpy as np
import matplotlib.pyplot as plt

uwall = dict(np.load('tauwall_target.npz'))
u60   = dict(np.load('tauwall_EWM60.npz'))

uwall_less = {}
u60_less = {}

   # campo_u_crop = campo_u[:(campo_u.shape[0] // 10)*10, : (campo_u.shape[1] // 10)*10]
   # campo_u_reshape = campo_u_crop.reshape(campo_u_crop.shape[0] // 10, 10, campo_u_crop.shape[1] // 10, 10)
   # campo_less_u = np.mean(campo_u_reshape, axis=(1,3))

#sottocampionamento con media
for (kw, ku) in zip(uwall.keys(), u60.keys()):
    wall = uwall[kw]
    wall = wall[:(wall.shape[0] // 10)*10, : (wall.shape[1] //10)*10]
    wall = wall.reshape(wall.shape[0] //10, 10, wall.shape[1] // 10, 10)
    wall = np.mean(wall,axis=(1,3))
    uwall_less[kw] = wall
    campo = u60[ku]
    campo = campo[:(campo.shape[0] // 10)*10, : (campo.shape[1] //10)*10]
    campo = campo.reshape(campo.shape[0] //10, 10, campo.shape[1] // 10, 10)
    campo = np.mean(campo,axis=(1,3))
    u60_less[ku] = campo

np.savez('tauwall_target_64x64.npz', **uwall_less)
np.savez('tauwall_EWM60_64x64.npz', **u60_less)
