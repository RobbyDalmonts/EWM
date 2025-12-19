import numpy as np

wall = dict(np.load('tauwall_target_64x64.npz'))
ewm = dict(np.load('tauwall_EWM60_64x64.npz'))
#uncond = dict(np.load('tauwall_uncond_all.npz'))
#ewm = np.load('tauwall_EWM60_64.npy')
#ewm = ewm[:27,:,:,:]

wall_array = np.empty([67,190,64,64])
ewm_array = np.empty([67,190,64,64])
alpha = 0.1388384832398331

for (ke,kw) in zip(ewm.keys(),wall.keys()): #uncond.keys():
    for i in range(ewm_array.shape[0]):
        for j in range(ewm_array.shape[1]):
            wall_array[i,j,:,:] = wall[kw]
            ewm_array[i,j,:,:] = ewm[ke] #alpha * ewm[i,j,:,:] + (1 - alpha) * uncond[ke]

np.save('tauwall_target_64.npy', wall_array)
np.save('tauwall_EWM60_64.npy', ewm_array)




#wall = dict(np.load('tauwall_target_64x64_shift.npz'))
#ewm = dict(np.load('tauwall_EWM60_64x64_shift.npz'))
##wall = dict(np.load('slice_u60_64x64.npz'))
#
#wall_array = np.empty([67,380,64,64])
#ewm_array = np.empty([67,380,64,64])
#
#for (kw,ke) in zip(wall.keys(), ewm.keys()):
#    for i in range(wall_array.shape[0]):
#        for j in range(wall_array.shape[1]):
#            wall_array[i,j,:,:] = wall[kw]
#            ewm_array[i,j,:,:] = ewm[ke]
#
#np.save('tauwall_target_64_shift.npy', wall_array)
#np.save('tauwall_EWM60_64_shift.npy', ewm_array)

