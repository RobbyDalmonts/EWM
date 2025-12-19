import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

linear = dict(np.load('linear_EWM_uncond.npz'))
dns = dict(np.load('tauwall_target_64x64_shift.npz'))
ewm = dict(np.load('tauwall_EWM60_64x64_shift.npz'))
unc = dict(np.load('unconditional.npz'))
alpha = 0.25370291
alpha = 0.5
for (kd,kl,ke,ku) in zip(dns.keys(),linear.keys(),ewm.keys(),unc.keys()):
    fig,ax = plt.subplots(2,4,figsize=(20,15))
    ax[0,0].imshow(dns[kd], cmap='coolwarm')
    ax[0,1].imshow(linear[kl], cmap='coolwarm')
    ax[0,2].imshow(alpha*ewm[ke], cmap='coolwarm')
    ax[0,3].imshow((1-alpha)*unc[ku], cmap='coolwarm')
    sns.histplot(dns[kd].ravel(), bins=100, kde=True, stat='density', ax= ax[1,0])
    ax[1,0].axvline(np.mean(dns[kd]), color='black')
    ax[1,0].text(
    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
    "MEAN: {:.4f}\nSTD: {:.4f}".format(np.mean(dns[kd]), np.std(dns[kd])),     # te
    transform=ax[1,0].transAxes,  # usa assi normalizzati (da 0 a 1)
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)

    sns.histplot(alpha*ewm[ke].ravel(), bins=100, kde=True, stat='density', ax= ax[1,2])
    ax[1,2].axvline(np.mean(alpha*ewm[ke]), color='black')
    ax[1,2].text(
    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
    "MEAN: {:.4f}\nSTD: {:.4f}".format(np.mean(alpha*ewm[ke]), np.std(alpha*ewm[ke])),     # te
    transform=ax[1,2].transAxes,  # usa assi normalizzati (da 0 a 1)
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)

    sns.histplot(linear[kl].ravel(), bins=100, kde=True, stat='density', ax= ax[1,1])
    ax[1,1].axvline(np.mean(linear[kl]), color='black')
    ax[1,1].text(
    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
    "MEAN: {:.4f}\nSTD: {:.4f}".format(np.mean(linear[kl]), np.std(linear[kl])),     # te
    transform=ax[1,1].transAxes,  # usa assi normalizzati (da 0 a 1)
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)

    sns.histplot((1-alpha)*unc[ku].ravel(), bins=100, kde=True, stat='density', ax= ax[1,3])
    ax[1,3].axvline(np.mean((1-alpha)*unc[ku]), color='black')
    ax[1,3].text(
    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
    "MEAN: {:.4f}\nSTD: {:.4f}".format(np.mean((1-alpha)*unc[ku]), np.std((1-alpha)*unc[ku])),     # te
    transform=ax[1,3].transAxes,  # usa assi normalizzati (da 0 a 1)
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)
    ax[0,0].set_title('dns')
    ax[0,1].set_title('linear')
    ax[0,2].set_title('alpha * ewm')
    ax[0,3].set_title('(1 - alpha) * unc')
    ax[1,0].set_yscale('log')
    ax[1,1].set_yscale('log')
    ax[1,2].set_yscale('log')
    ax[1,3].set_yscale('log')
    ax[1,0].set_xlim(dns[kd].min(), dns[kd].max())
    ax[1,1].set_xlim(dns[kd].min(), dns[kd].max())
    ax[1,2].set_xlim(dns[kd].min(), dns[kd].max())
    ax[1,3].set_xlim(dns[kd].min(), dns[kd].max())
    plt.savefig('sample_image/confronto_{}'.format(ku))


    
