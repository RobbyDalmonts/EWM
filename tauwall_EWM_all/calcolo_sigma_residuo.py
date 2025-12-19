import numpy as np

dns = np.load('tauwall_target_64.npy')
ewm = np.load('tauwall_EWM60_64.npy')

dns_flatten = dns.ravel()
ewm_flatten = ewm.ravel()


