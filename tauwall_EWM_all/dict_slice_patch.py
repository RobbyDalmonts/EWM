import numpy as np

u_wall = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/tauwall_EWM_all/tauwall_target.npz'))
#u_wall = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/tauwall_EWM_all/tauwall_EWM120.npz'))
#u_10   = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u10.npz'))
#u_15   = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u15.npz'))
#u_20   = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u20.npz'))
#u_30   = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u30.npz'))
#u_40   = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u40.npz'))
#u_50   = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u50.npz'))
#u_60   = dict(np.load('/leonardo_work/IscrC_MLWM-CF/last_EWM/tauwall_EWM_all/slice_u60.npz'))
#u_70   = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u70.npz'))
#u_80   = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u80.npz'))
#u_90   = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u90.npz'))
#u_100  = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u100.npz'))
#u_110  = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u110.npz'))
#u_120  = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u120.npz'))
#u_130  = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u130.npz'))
#u_140  = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u140.npz'))
#u_150  = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u150.npz'))
#u_160  = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u160.npz'))
#u_170  = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u170.npz'))
#u_180  = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u180.npz'))
#u_190  = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u190.npz'))
#u_200  = dict(np.load('/leonardo_work/IscrC_MLWM-CF/slice_Retau1000_all/slice_u200.npz'))

u_10   = dict(np.load('EWM_slice_match/tauwall_EWM10.npz'))
u_15   = dict(np.load('EWM_slice_match/tauwall_EWM15.npz'))
u_20   = dict(np.load('EWM_slice_match/tauwall_EWM20.npz'))
u_30   = dict(np.load('EWM_slice_match/tauwall_EWM30.npz'))
u_40   = dict(np.load('EWM_slice_match/tauwall_EWM40.npz'))
u_50   = dict(np.load('EWM_slice_match/tauwall_EWM50.npz'))
u_60   = dict(np.load('EWM_slice_match/tauwall_EWM60.npz'))
u_70   = dict(np.load('EWM_slice_match/tauwall_EWM70.npz'))
u_80   = dict(np.load('EWM_slice_match/tauwall_EWM80.npz'))
u_90   = dict(np.load('EWM_slice_match/tauwall_EWM90.npz'))
u_100  = dict(np.load('EWM_slice_match/tauwall_EWM100.npz'))
u_110  = dict(np.load('EWM_slice_match/tauwall_EWM110.npz'))
u_120  = dict(np.load('EWM_slice_match/tauwall_EWM120.npz'))
u_130  = dict(np.load('EWM_slice_match/tauwall_EWM130.npz'))
u_140  = dict(np.load('EWM_slice_match/tauwall_EWM140.npz'))
u_150  = dict(np.load('EWM_slice_match/tauwall_EWM150.npz'))
u_160  = dict(np.load('EWM_slice_match/tauwall_EWM160.npz'))
u_170  = dict(np.load('EWM_slice_match/tauwall_EWM170.npz'))
u_180  = dict(np.load('EWM_slice_match/tauwall_EWM180.npz'))
u_190  = dict(np.load('EWM_slice_match/tauwall_EWM190.npz'))
u_200  = dict(np.load('EWM_slice_match/tauwall_EWM200.npz'))

#sample_10 = {}
#sample_60 = {}
#res = 64 #resolution
#nx = 1216
#ny = 640
#campo_10_shift = np.zeros((1216,640))
#campo_60_shift = np.zeros((1216,640))
#sample_10 = {}
#sample_60 = {}
#res = 64 #resolution
#nx = 1216
#ny = 640
#campo_10_shift = np.zeros((1216,640))
#campo_60_shift = np.zeros((1216,640))
#for (k10,k60) in zip(u_wall.keys(),u_60.keys()):
#    campo_10 = u_wall[k10]
#    campo_60 = u_60[k60]
#    campo_10_shift[:48,:] = campo_10[-48:,:]   #prendo i batch lungo x (scorro per righe)
#    campo_10_shift[48:,:] = campo_10[:-48,:]
#    campo_60_shift[:48,:] = campo_60[-48:,:]
#    campo_60_shift[48:,:] = campo_60[:-48,:]
#    i = 0
#    for k in range(int(ny/res)):
#        for j in range(int(nx/res)):
#            i = i + 1
#            c10 = campo_10[res*j:res*j+res, res*k:res*k+res]
#            c10_shift = campo_10_shift[res*j:res*j+res, res*k:res*k+res]
#            c60 = campo_60[res*j:res*j+res, res*k:res*k+res]
#            c60_shift = campo_60_shift[res*j:res*j+res, res*k:res*k+res]
#
#            key10 = '{}_{}'.format(k10,i)
#            key10_shift = '{}_{}_shift'.format(k10,i)
#            key60 = '{}_{}'.format(k60,i)
#            key60_shift = '{}_{}_shift'.format(k60,i)
#
#            sample_10[key10] = c10
#            sample_10[key10_shift] = c10_shift
#            sample_60[key60] = c60
#            sample_60[key60_shift] = c60_shift
#
#
#np.savez('tauwall_target_64x64_shift.npz',**sample_10)
#np.savez('tauwall_EWM60_64x64_shift.npz',**sample_60)
#
#print(len(sample_10.keys()))
#print(len(sample_60.keys()))

#sample_10 = {}
#sample_60 = {}
res = 64 #resolution
nx = 1216
ny = 640
#
#target = np.zeros((67,190,res,res))
#ewm = np.zeros((67,190,res,res))
#n=0
#for (k10,k60) in zip(u_wall.keys(),u_60.keys()):
#    campo_10 = u_wall[k10]
#    campo_60 = u_60[k60]
#    i = 0
#    for j in range(int(nx/res)):
#        for k in range(int(ny/res)):
#            target[n][i] = campo_10[res*j:res*j+res, res*k:res*k+res]
#            ewm[n][i] = campo_60[res*j:res*j+res, res*k:res*k+res]
#            i = i + 1
#    n = n + 1
#
#np.save('tauwall_target_64.npy',target)
#np.save('slice_u120_64.npy',ewm)
#exit()
    

target = np.zeros((67,190,res,res))
u10 = np.zeros((67,190,res,res))
u15 = np.zeros((67,190,res,res))
u20 = np.zeros((67,190,res,res))
u30 = np.zeros((67,190,res,res))
u40 = np.zeros((67,190,res,res))
u50 = np.zeros((67,190,res,res))
u60 = np.zeros((67,190,res,res))
u70 = np.zeros((67,190,res,res))
u80 = np.zeros((67,190,res,res))
u90 = np.zeros((67,190,res,res))
u100 = np.zeros((67,190,res,res))
u110 = np.zeros((67,190,res,res))
u120 = np.zeros((67,190,res,res))
u130 = np.zeros((67,190,res,res))
u140 = np.zeros((67,190,res,res))
u150 = np.zeros((67,190,res,res))
u160 = np.zeros((67,190,res,res))
u170 = np.zeros((67,190,res,res))
u180 = np.zeros((67,190,res,res))
u190 = np.zeros((67,190,res,res))
u200 = np.zeros((67,190,res,res))
n=0
for (kw,k10,k15,k20,k30,k40,k50,k60,k70,k80,k90,k100,k110,k120,k130,k140,k150,k160,k170,k180,k190,k200) in zip(u_wall.keys(),u_10.keys(),u_15.keys(),u_20.keys(),u_30.keys(),u_40.keys(),u_50.keys(),u_60.keys(),u_70.keys(),u_80.keys(),u_90.keys(),u_100.keys(),u_110.keys(),u_120.keys(),u_130.keys(),u_140.keys(),u_150.keys(),u_160.keys(),u_170.keys(),u_180.keys(),u_190.keys(),u_200.keys()):
    campo_w = u_wall[kw]
    campo_10 = u_10[k10]
    campo_15 = u_15[k15]
    campo_20 = u_20[k20]
    campo_30 = u_30[k30]
    campo_40 = u_40[k40]
    campo_50 = u_50[k50]
    campo_60 = u_60[k60]
    campo_70 = u_70[k70]
    campo_80 = u_80[k80]
    campo_90 = u_90[k90]
    campo_100 = u_100[k100]
    campo_110 = u_110[k110]
    campo_120 = u_120[k120]
    campo_130 = u_130[k130]
    campo_140 = u_140[k140]
    campo_150 = u_150[k150]
    campo_160 = u_160[k160]
    campo_170 = u_170[k170]
    campo_180 = u_180[k180]
    campo_190 = u_190[k190]
    campo_200 = u_200[k200]
    i = 0
    for j in range(int(nx/res)):
        for k in range(int(ny/res)):
            target[n][i] = campo_w[res*j:res*j+res, res*k:res*k+res]
            u10[n][i] = campo_10[res*j:res*j+res, res*k:res*k+res]
            u15[n][i] = campo_15[res*j:res*j+res, res*k:res*k+res]
            u20[n][i] = campo_20[res*j:res*j+res, res*k:res*k+res]
            u30[n][i] = campo_30[res*j:res*j+res, res*k:res*k+res]
            u40[n][i] = campo_40[res*j:res*j+res, res*k:res*k+res]
            u50[n][i] = campo_50[res*j:res*j+res, res*k:res*k+res]
            u60[n][i] = campo_60[res*j:res*j+res, res*k:res*k+res]
            u70[n][i] = campo_70[res*j:res*j+res, res*k:res*k+res]
            u80[n][i] = campo_80[res*j:res*j+res, res*k:res*k+res]
            u90[n][i] = campo_90[res*j:res*j+res, res*k:res*k+res]
            u100[n][i] = campo_100[res*j:res*j+res, res*k:res*k+res]
            u110[n][i] = campo_110[res*j:res*j+res, res*k:res*k+res]
            u120[n][i] = campo_120[res*j:res*j+res, res*k:res*k+res]
            u130[n][i] = campo_130[res*j:res*j+res, res*k:res*k+res]
            u140[n][i] = campo_140[res*j:res*j+res, res*k:res*k+res]
            u150[n][i] = campo_150[res*j:res*j+res, res*k:res*k+res]
            u160[n][i] = campo_160[res*j:res*j+res, res*k:res*k+res]
            u170[n][i] = campo_170[res*j:res*j+res, res*k:res*k+res]
            u180[n][i] = campo_180[res*j:res*j+res, res*k:res*k+res]
            u190[n][i] = campo_190[res*j:res*j+res, res*k:res*k+res]
            u200[n][i] = campo_200[res*j:res*j+res, res*k:res*k+res]
            i = i + 1
    n = n + 1

np.save('EWM_sample_match/tauwall_target_64.npy',target)
np.save('EWM_sample_match/tauwall_EWM10_64.npy',u10)
np.save('EWM_sample_match/tauwall_EWM15_64.npy',u15)
np.save('EWM_sample_match/tauwall_EWM20_64.npy',u20)
np.save('EWM_sample_match/tauwall_EWM30_64.npy',u30)
np.save('EWM_sample_match/tauwall_EWM40_64.npy',u40)
np.save('EWM_sample_match/tauwall_EWM50_64.npy',u50)
np.save('EWM_sample_match/tauwall_EWM60_64.npy',u60)
np.save('EWM_sample_match/tauwall_EWM70_64.npy',u70)
np.save('EWM_sample_match/tauwall_EWM80_64.npy',u80)
np.save('EWM_sample_match/tauwall_EWM90_64.npy',u90)
np.save('EWM_sample_match/tauwall_EWM100_64.npy',u100)
np.save('EWM_sample_match/tauwall_EWM110_64.npy',u110)
np.save('EWM_sample_match/tauwall_EWM120_64.npy',u120)
np.save('EWM_sample_match/tauwall_EWM130_64.npy',u130)
np.save('EWM_sample_match/tauwall_EWM140_64.npy',u140)
np.save('EWM_sample_match/tauwall_EWM150_64.npy',u150)
np.save('EWM_sample_match/tauwall_EWM160_64.npy',u160)
np.save('EWM_sample_match/tauwall_EWM170_64.npy',u170)
np.save('EWM_sample_match/tauwall_EWM180_64.npy',u180)
np.save('EWM_sample_match/tauwall_EWM190_64.npy',u190)
np.save('EWM_sample_match/tauwall_EWM200_64.npy',u200)

exit()



for (k10,k60) in zip(u_wall.keys(),u_60.keys()):
    campo_10 = u_wall[k10]
    campo_60 = u_60[k60]
    i = 0
    for j in range(int(nx/res)):
        for k in range(int(ny/res)):
            i = i + 1
            c10 = campo_10[res*j:res*j+res, res*k:res*k+res]
            c60 = campo_60[res*j:res*j+res, res*k:res*k+res]

            key10 = '{}_{}'.format(k10,i)
            key60 = '{}_{}'.format(k60,i)

            sample_10[key10] = c10
            sample_60[key60] = c60


np.savez('tauwall_target_64x64.npz',**sample_10)
np.savez('tauwall_EWM60_64x64.npz',**sample_60)
#for i in range(1,68):
#    for j in range(1,191):
#        sample_10 = np.load('u_10_{}_{}.npy'.format(i,j))
#        sample_30 = np.load('u_30_{}_{}.npy'.format(i,j))
#        sample_60 = np.load('u_60_{}_{}.npy'.format(i,j))
#        sample_120 = np.load('u_120_{}_{}.npy'.format(i,j))
#
#        k10 = 'u_10_{}_{}'.format(i,j)
#        k30 = 'u_30_{}_{}'.format(i,j)
#        k60 = 'u_60_{}_{}'.format(i,j)
#        k120 = 'u_120_{}_{}'.format(i,j)
#
#        u10[k10] = sample_10
#        u30[k30] = sample_30
#        u60[k60] = sample_60
#        u120[k120] = sample_120

