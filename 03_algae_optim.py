# %%
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import GPy
import h5py
from scipy.stats import norm
from GPy.models import GPRegression
from warnings import catch_warnings
from warnings import simplefilter
from algae_common import *

run_dir = '/dev/shm/calbert/vecma21'

os.makedirs(os.path.join(run_dir, 'RESULTS'), exist_ok=True)
shutil.copytree(os.path.join(template_dir, 'INPUT_DATA'),
    os.path.join(run_dir, 'INPUT_DATA'), symlinks=False, dirs_exist_ok=True)
#%%
nt = len(yref)
plt.figure()
plt.plot(yref)

# %%

X = np.load('Xtrain0.npy')
y = cost_y(np.load('ytrain0.npy'))

model = GPy.models.GPRegression(X, y.reshape(-1, 1), k,
        noise_var=1e-4, mean_function=mf)

with h5py.File('sur0.hdf5', 'r') as f:
    model.param_array[:] = f['param_array']

print(model)
print(model.kern.lengthscale)
#%%

ymin = np.array([np.min(y)])

def surrogate(model: GPRegression, X):
    with catch_warnings():
        simplefilter("ignore")
        mu, var = model.predict(X, full_cov=False)
        return mu, np.sqrt(var)

def acquisition(x, model):
    mu, std = surrogate(model, x.reshape(-1, nvar))
    std = std + 1e-31
    probs = (ymin[-1] - mu)*norm.cdf(ymin[-1], mu, std) + \
        std**2 * norm.pdf(ymin[-1], mu, std)
    return -probs

def opt_acquisition(X, model):
    Xsamples = np.random.rand(4096, nvar)
    scores = acquisition(Xsamples, model)
    ix = np.argmin(scores)
    return Xsamples[ix, :]

for i in range(nsamp0):
    x = opt_acquisition(X, model)
    ytrue = cost(box_to_actual(x), run_dir)
    yest, _ = surrogate(model, x.reshape(-1, nvar))
    print(yest[0, 0], ytrue)
    # add the data to the dataset
    X = np.vstack((X, x))
    y = np.append(y, ytrue)
    ymin = np.append(ymin, np.min(y))
    # update the model
    model.set_XY(X, y.reshape(-1, 1))
    model.optimize('bfgs')

kopt = np.argmin(y)
xopt = X[kopt,:]
# %%
plt.figure()
for k in range(len(X)):
    xt = box_to_actual(X[k, :])
    ypred, yvar = model.predict(actual_to_box(xt).reshape(-1, nvar))
    plt.semilogy([k, k], [ypred[0,0], ypred[0,0] + 2*np.sqrt(yvar[0,0])], 'k-')

# %%
plt.figure()
plt.scatter(X[:,0], X[:,1], c=model.predict(X)[0])
plt.colorbar()

plt.figure()
XN = box_to_actual(X)
plt.scatter(XN[:,0], XN[:,1], c=model.predict(X)[0])
plt.colorbar()


# %%
plt.figure()
plt.plot(yref)
plt.plot(blackbox(mean, run_dir), '-')
plt.plot(blackbox(box_to_actual(xopt), run_dir), '--')
plt.legend(['reference', 'start', 'optimized'])

# %%
plt.figure()
plt.semilogy(ymin)

# %%
ropt = mean
ropt[:nvar] = box_to_actual(xopt)
np.save('ropt', ropt)

np.save('Xtrain1.npy', X.reshape(2*nsamp0, nvar))
np.save('ytrain1.npy', y.reshape(2*nsamp0, -1))
model.save('sur1.hdf5')

# %%
