import numpy as np
import matplotlib.pyplot as plt
from algae_common import *

nproc = 16
nwarm = 250
nmc = 1000

x = []
for kproc in range(nproc):
    x.append(np.load(f'xmc_{kproc}.npy'))

means = [np.mean(xk[nwarm+1:], 0) for xk in x]
covs = [np.cov(xk[nwarm+1:].T) for xk in x]
stds = [np.sqrt(np.diag(c)) for c in covs]

plt.figure()
for m in means:
    plt.plot(m[0], m[1], 'x')

#%%

x = []
for kproc in range(nproc):
    x.append(np.load(f'xmc_da_{kproc}.npy'))

means = [np.mean(xk[nwarm+1:], 0) for xk in x]
covs = [np.cov(xk[nwarm+1:].T) for xk in x]
stds = [np.sqrt(np.diag(c)) for c in covs]

plt.figure()
for m in means:
    plt.plot(m[0], m[1], 'x')
