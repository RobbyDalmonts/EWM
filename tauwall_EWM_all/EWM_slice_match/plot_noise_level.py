import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dns = dict(np.load('tauwall_target.npz'))
e10 = dict(np.load('tauwall_EWM10.npz'))
e15 = dict(np.load('tauwall_EWM15.npz'))
e20 = dict(np.load('tauwall_EWM20.npz'))
e30 = dict(np.load('tauwall_EWM30.npz'))
e40 = dict(np.load('tauwall_EWM40.npz'))
e50 = dict(np.load('tauwall_EWM50.npz'))
e60 = dict(np.load('tauwall_EWM60.npz'))
e70 = dict(np.load('tauwall_EWM70.npz'))
e80 = dict(np.load('tauwall_EWM80.npz'))
e90 = dict(np.load('tauwall_EWM90.npz'))
e100 = dict(np.load('tauwall_EWM100.npz'))
e110 = dict(np.load('tauwall_EWM110.npz'))
e120 = dict(np.load('tauwall_EWM120.npz'))
e130 = dict(np.load('tauwall_EWM130.npz'))
e140 = dict(np.load('tauwall_EWM140.npz'))
e150 = dict(np.load('tauwall_EWM150.npz'))
e160 = dict(np.load('tauwall_EWM160.npz'))
e170 = dict(np.load('tauwall_EWM170.npz'))
e180 = dict(np.load('tauwall_EWM180.npz'))
e190 = dict(np.load('tauwall_EWM190.npz'))
e200 = dict(np.load('tauwall_EWM200.npz'))

attenuation = np.load('attenuation_signal.npy')
noise_level = np.load('sigma_DDIM_EWM.npy')
t = np.linspace(1,1000,1000)

y_list = ['dns',10,15,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]

dns_l = []
e10_l = []
e15_l = []
e20_l = []
e30_l = []
e40_l = []
e50_l = []
e60_l = []
e70_l = []
e80_l = []
e90_l = []
e100_l = []
e110_l = []
e120_l = []
e130_l = []
e140_l = []
e150_l = []
e160_l = []
e170_l = []
e180_l = []
e190_l = []
e200_l = []

for (kd,k10,k15,k20,k30,k40,k50,k60,k70,k80,k90,k100,k110,k120,k130,k140,k150,k160,k170,k180,k190,k200) in zip(dns.keys(),e10.keys(),e15.keys(),e20.keys(),e30.keys(),e40.keys(),e50.keys(),e60.keys(),e70.keys(),e80.keys(),e90.keys(),e100.keys(),e110.keys(),e120.keys(),e130.keys(),e140.keys(),e150.keys(),e160.keys(),e170.keys(),e180.keys(),e190.keys(),e200.keys()):
    dns_l.append(dns[kd].ravel())
    e10_l.append(e10[k10].ravel())
    e15_l.append(e15[k15].ravel())
    e20_l.append(e20[k20].ravel())
    e30_l.append(e30[k30].ravel())
    e40_l.append(e40[k40].ravel())
    e50_l.append(e50[k50].ravel())
    e60_l.append(e60[k60].ravel())
    e70_l.append(e70[k70].ravel())
    e80_l.append(e80[k80].ravel())
    e90_l.append(e90[k90].ravel())
    e100_l.append(e100[k100].ravel())
    e110_l.append(e110[k110].ravel())
    e120_l.append(e120[k120].ravel())
    e130_l.append(e130[k130].ravel())
    e140_l.append(e140[k140].ravel())
    e150_l.append(e150[k150].ravel())
    e160_l.append(e160[k160].ravel())
    e170_l.append(e170[k170].ravel())
    e180_l.append(e180[k180].ravel())
    e190_l.append(e190[k190].ravel())
    e200_l.append(e200[k200].ravel())

dns_l = np.array(dns_l)
e10_l = np.array(e10_l)
e15_l = np.array(e15_l)
e20_l = np.array(e20_l)
e30_l = np.array(e30_l)
e40_l = np.array(e40_l)
e50_l = np.array(e50_l)
e60_l = np.array(e60_l)
e70_l = np.array(e70_l)
e80_l = np.array(e80_l)
e90_l = np.array(e90_l)
e100_l = np.array(e100_l)
e110_l = np.array(e110_l)
e120_l = np.array(e120_l)
e130_l = np.array(e130_l)
e140_l = np.array(e140_l)
e150_l = np.array(e150_l)
e160_l = np.array(e160_l)
e170_l = np.array(e170_l)
e180_l = np.array(e180_l)
e190_l = np.array(e190_l)
e200_l = np.array(e200_l)

dns_mean = np.mean(dns_l)
e10_mean = np.mean(e10_l)
e15_mean = np.mean(e15_l)
e20_mean = np.mean(e20_l)
e30_mean = np.mean(e30_l)
e40_mean = np.mean(e40_l)
e50_mean = np.mean(e50_l)
e60_mean = np.mean(e60_l)
e70_mean = np.mean(e70_l)
e80_mean = np.mean(e80_l)
e90_mean = np.mean(e90_l)
e100_mean = np.mean(e100_l)
e110_mean = np.mean(e110_l)
e120_mean = np.mean(e120_l)
e130_mean = np.mean(e130_l)
e140_mean = np.mean(e140_l)
e150_mean = np.mean(e150_l)
e160_mean = np.mean(e160_l)
e170_mean = np.mean(e170_l)
e180_mean = np.mean(e180_l)
e190_mean = np.mean(e190_l)
e200_mean = np.mean(e200_l)

dns_std = np.std(dns_l)
e10_std = np.std(e10_l)
e15_std = np.std(e15_l)
e20_std = np.std(e20_l)
e30_std = np.std(e30_l)
e40_std = np.std(e40_l)
e50_std = np.std(e50_l)
e60_std = np.std(e60_l)
e70_std = np.std(e70_l)
e80_std = np.std(e80_l)
e90_std = np.std(e90_l)
e100_std = np.std(e100_l)
e110_std = np.std(e110_l)
e120_std = np.std(e120_l)
e130_std = np.std(e130_l)
e140_std = np.std(e140_l)
e150_std = np.std(e150_l)
e160_std = np.std(e160_l)
e170_std = np.std(e170_l)
e180_std = np.std(e180_l)
e190_std = np.std(e190_l)
e200_std = np.std(e200_l)

#dns_l = (dns_l - dns_mean) / dns_std
#e10_l = (e10_l - dns_mean) / dns_std
#e15_l = (e15_l - dns_mean) / dns_std
#e20_l = (e20_l - dns_mean) / dns_std
#e30_l = (e30_l - dns_mean) / dns_std
#e40_l = (e40_l - dns_mean) / dns_std
#e50_l = (e50_l - dns_mean) / dns_std
#e60_l = (e60_l - dns_mean) / dns_std
#e70_l = (e70_l - dns_mean) / dns_std
#e80_l = (e80_l - dns_mean) / dns_std
#e90_l = (e90_l - dns_mean) / dns_std
#e100_l = (e100_l - dns_mean) / dns_std
#e110_l = (e110_l - dns_mean) / dns_std
#e120_l = (e120_l - dns_mean) / dns_std
#e130_l = (e130_l - dns_mean) / dns_std
#e140_l = (e140_l - dns_mean) / dns_std
#e150_l = (e150_l - dns_mean) / dns_std
#e160_l = (e160_l - dns_mean) / dns_std
#e170_l = (e170_l - dns_mean) / dns_std
#e180_l = (e180_l - dns_mean) / dns_std
#e190_l = (e190_l - dns_mean) / dns_std
#e200_l = (e200_l - dns_mean) / dns_std


standard_data = [dns_l,e10_l,e15_l,e20_l,e30_l,e40_l,e50_l,e60_l,e70_l,e80_l,e90_l,e100_l,e110_l,e120_l,e130_l,e140_l,e150_l,e160_l,e170_l,e180_l,e190_l,e200_l]

standard_data = [dns_l, e10_l, e30_l, e60_l, e100_l, e160_l]
y_list = ['dns',10,60,120,200]

max_point = 500000
fig,ax = plt.subplots(figsize=(10,10))
for (i,label) in zip(standard_data,y_list):
    if i.shape[0] > max_point:
        data_sample = np.random.choice(i.ravel(),max_point,replace=False)
    else:
        data_sample = i.ravel()

    if label == 'dns':
        sns.kdeplot(data_sample,linewidth=2.5,color='black',linestyle='--',label=str(label),ax=ax)
    else:
        sns.kdeplot(data_sample,linewidth=2.5,label=str(label),ax=ax)
    
ax.legend()
#ax.set_xlim(-5,6)
ax.set_xlim(0,0.007)
ax.set_title('pdf dist')
plt.savefig('pdf_less_less_nonnorm.png',dpi=300)
plt.close()
exit()
mean_standard = []
for i in standard_data:
    mean_standard.append(np.mean(i))
    print(np.mean(i))

std_standard = []
for i in standard_data:
    std_standard.append(1 - np.std(i))  #####

idx = []
for i in std_standard:
    idx.append(np.abs(noise_level - i).argmin())

t_inv = np.linspace(1000,1,1000)
fig,ax = plt.subplots(figsize=(10,10))
ax.plot(t,noise_level,color='black',label='noise_level')   ####
for (i,std,label) in zip(idx,std_standard,y_list):
    ax.scatter(t[i],std,label='y+={}'.format(label))      #### t_inv

plt.legend()
plt.savefig('noise_level_update.png',dpi=300)
plt.close()

for (i,std,label) in zip(idx,std_standard,y_list):
    print('y+ = {} : timestep = {}, noise_level = {}'.format(label,i,std))####format(label,1000-i,1-std))

std_standard = np.array(std_standard)

std_noise = std_standard
attenuation_e = std_noise**2
attenuation_e = 1 - attenuation_e
attenuation_e = np.sqrt(attenuation_e)
fig,ax = plt.subplots(figsize=(10,10))
ax.plot(t,attenuation,color='black',label='attenuation')   ####
for (i,att,label) in zip(idx,attenuation_e,y_list):
    ax.scatter(t[i],att,label='y+={}'.format(label))  ####t_inv

plt.legend()
plt.savefig('attenuation_level_update.png',dpi=300)
plt.close()

print('-----------------------------------------------------------------------------------------------------------------------------------------------')
for (i,att,label) in zip(idx,attenuation_e,y_list):
    print('y+ = {} : timestep = {}, attenuation_level = {}'.format(label,i,att))

np.save('y_list.npy',y_list[1:])
timestep = []
noise_level = []
for (i,std) in zip(idx,std_standard):
    timestep.append(t[i])
    noise_level.append(std)
np.save('noise_level_update.npy',noise_level)
np.save('timesteps_update.npy',timestep)
