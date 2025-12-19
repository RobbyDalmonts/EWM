import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sample_dns = np.load('sample_dns_t.npy')  #già scalati

sample_ewm = []
campo = np.load('tauwall_EWM10_64.npy')
campo = campo[:3,...]
sample_ewm.append(campo)
campo = np.load('tauwall_EWM15_64.npy')
campo = campo[:3,...]
sample_ewm.append(campo)


match = np.linspace(20,200,19, dtype = np.int64)
for i in match:
    campo = np.load('tauwall_EWM{}_64.npy'.format(i))  #da scalare
    campo = campo[:3,...]
    sample_ewm.append(campo)

sample_ewm = np.array(sample_ewm)

print(sample_dns.shape)
print(sample_ewm.shape)

dns_mean = 0.00237064671915439   #tau_less statistics. solo i (3,190,64,64)
dns_std = 0.0009744609912060021

sample_ewm = (sample_ewm - np.mean(sample_ewm)) / (np.std(sample_ewm) + 1e-12) * dns_std + dns_mean

att = np.array([0.99820554, 0.9902758 , 0.9756261 , 0.94309527, 0.91868824,
       0.90121585, 0.88787454, 0.87813115, 0.8695259 , 0.8621823 ,
       0.8561982 , 0.85011953, 0.8455002 , 0.84083027, 0.8361109 ,
       0.8313432 , 0.8281384 , 0.8249131 , 0.8216674 , 0.8184017 ,
       0.81511647])

e = np.random.randn(3,190,64,64)

sigma = np.array([0.05988068, 0.13911796, 0.21943937, 0.33252257, 0.39498338,
       0.43337044, 0.46008566, 0.47841996, 0.4938873 , 0.50659806,
       0.5166475 , 0.52658975, 0.5339751 , 0.5412989 , 0.5485605 ,
       0.55575943, 0.5605236 , 0.56525964, 0.56996727, 0.5746466 ,
       0.5792971 ])

for (i,a,s) in zip(range(sample_ewm.shape[0]),reversed(att),reversed(sigma)):
    sample_ewm[i] = a * sample_ewm[i] + s * e

step = np.array([14,  39,  65, 103, 125, 139, 149, 156, 162, 167, 171, 175,
    178, 181, 184, 187, 189, 191, 193, 195, 197])
y_list = [10,15,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
for (i,t,label) in zip(range(sample_ewm.shape[0]),step,y_list):
    fig,ax = plt.subplots(figsize=(13,13))
    sns.histplot(sample_dns[i+1,...].ravel(), stat='density', bins=150, kde=True)#,ax=ax[0])
    sns.histplot(sample_ewm[i,...].ravel(), stat='density', bins=150, color='red', kde=True)#,ax=ax[1])
    #ax.set_xlim(sample_dns[i+1,...].min(),sample_dns[i+1,...].max())
    #ax[1].set_xlim(sample_dns[i+1].min(),sample_dns[i+1].max())
    ax.set_yscale('log')
    #ax[1].set_yscale('log')
    ax.set_ylim(bottom=1e-6)
    #ax[1].set_ylim(bottom=1e-6)
    ax.axvline(np.mean(sample_dns[i+1,...]), color='black',label='mean_dns')
    ax.axvline(np.mean(sample_ewm[i,...]), color='green',label='mean_ewm')
    ax.set_title('t = {} y = {}. Mean_dns = {:.5f}, Mean_ewm = {:.5f}, std_dns = {:.5f}, std_ewm = {:.5f}'.format(t,label,np.mean(sample_dns[i+1,...]),np.mean(sample_ewm[i,...]),np.std(sample_dns[i+1,...]),np.std(sample_ewm[i,...])))
    #ax[0].set_title('sample dns t = {}. Mean = {:.5f}, std = {:.5f}'.format(t,np.mean(sample_dns[i+1]),np.std(sample_dns[i+1])))
    #ax[1].set_title('sample ewm y = {}. Mean = {:.5f}, std = {:.5f}'.format(label,np.mean(sample_ewm[i]),np.std(sample_ewm[i])))
    plt.legend()
    plt.tight_layout()
    plt.savefig('pdf_figure_noised/pdf_sample_ewm_scaled/pdf_y{}_t{}_ewm_prescaled.png'.format(label,t))
    plt.close()




