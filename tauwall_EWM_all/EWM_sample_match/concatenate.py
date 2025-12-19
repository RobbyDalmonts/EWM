import numpy as np

#sample_list = []
#sample_list.append('tauwall_EWM10_64.npy')
#sample_list.append('tauwall_EWM15_64.npy')
#
#match = np.linspace(20,200,19)
#for i in match:
#    sample_list.append('tauwall_EWM{}_64.npy'.format(int(i)))
#
##carico primo sample per leggere shape e dtype
#e10 = np.load(sample_list[0])
#n_per_sample = e10.shape[0]
#final_shape = (len(sample_list) * n_per_sample, *e10.shape[1:])
#
##Preallocazione
#final_array = np.empty(final_shape, dtype=e10.dtype)
#
##Copia sicura file per file
#final_array[:n_per_sample] = e10
#
#for idx,fname in enumerate(sample_list[1:], start=1):
#    arr = np.load(fname)
#    final_array[idx*n_per_sample:(idx+1)*n_per_sample] = arr
#    del arr #libera Ram subito
#
#np.save('all_match_64.npy',final_array)
#print(final_array.shape)
#
sample_list = []

i = 0
while True:
    sample_list.append('tauwall_target_64.npy')
    i +=1 
    if i >= 21:
        break


#carico primo sample per leggere shape e dtype
e10 = np.load(sample_list[0])
n_per_sample = e10.shape[0]
final_shape = (len(sample_list) * n_per_sample, *e10.shape[1:])

#Preallocazione
final_array = np.empty(final_shape, dtype=e10.dtype)

#Copia sicura file per file
final_array[:n_per_sample] = e10

for idx,fname in enumerate(sample_list[1:], start=1):
    arr = np.load(fname)
    final_array[idx*n_per_sample:(idx+1)*n_per_sample] = arr
    del arr #libera Ram subito

np.save('all_target_64.npy',final_array)
print(final_array.shape)
sample_list = []
#i=0
#whil

