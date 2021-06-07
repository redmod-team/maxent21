# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from algae_common import *

acor = 10

nwarm = 100
nmc = 1000

x = []
for kproc in range(0, 4):
    x.append(np.load(f'xmc_{kproc}.npy'))

means = [np.mean(xk[nwarm+1:], 0) for xk in x]
covs = [np.cov(xk[nwarm+1:].T) for xk in x]
stds = [np.sqrt(np.diag(c)) for c in covs]

plt.figure()
for m in means:
    plt.plot(m[0], m[1], 'x')
#%%
bins = [np.linspace(0, 400, 21), np.linspace(1, 3, 21)]

X = np.vstack([xk[nwarm+1:,:] for xk in x])


plt.figure(figsize=[4.0, 2.0])
pd.plotting.autocorrelation_plot(X[:10000, 0])
pd.plotting.autocorrelation_plot(X[:10000, 1], linestyle='dashed')
plt.xlim(0, 1000)
plt.tight_layout()
plt.savefig('paper/fig/acor_algae.pdf')

plt.figure(figsize=(4.0, 4.0))
sns.displot(x=X[::acor,0], y=X[::acor,1], height=4, bins=bins)
plt.plot(X[::acor,0], X[::acor,1], 'k,', alpha=0.2)
plt.xlim(0, 400)
plt.ylim(1.0, 3.0)
plt.xlabel(r'$K_{\mathrm{light}}$')
plt.ylabel(r'$\mu_0$')
plt.tight_layout()
plt.savefig('paper/fig/mcmc_algae.pdf')

# %%

x = []
for kproc in range(4, 8):
    x.append(np.load(f'xmc_da_{kproc}.npy'))

means = [np.mean(xk[nwarm+1:], 0) for xk in x]
covs = [np.cov(xk[nwarm+1:].T) for xk in x]
stds = [np.sqrt(np.diag(c)) for c in covs]

plt.figure()
for m in means:
    plt.plot(m[0], m[1], 'x')
#%%
X = np.vstack([xk[nwarm+1:,:] for xk in x])

plt.figure(figsize=[4.0, 2.0])
pd.plotting.autocorrelation_plot(X[:, 0])
pd.plotting.autocorrelation_plot(X[:, 1], linestyle='dashed')
plt.xlim(0, 1000)
plt.tight_layout()
plt.savefig('paper/fig/acor_algae_da.pdf')

plt.figure(figsize=(4.0, 4.0))
sns.displot(x=X[::acor,0], y=X[::acor,1], height=4, bins=bins)
plt.plot(X[::acor,0], X[::acor,1], 'k,', alpha=0.2)
plt.xlim(0, 400)
plt.ylim(1.0, 3.0)
plt.xlabel(r'$K_{\mathrm{light}}$')
plt.ylabel(r'$\mu_0$')
plt.tight_layout()
plt.savefig('paper/fig/mcmc_algae_da.pdf')

# %%
