# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import profit
from subprocess import run
from scipy.optimize import leastsq
from scipy.special import erfinv, erf
from scipy.stats import norm
from profit.util.halton import halton

template_dir = os.path.expanduser('~/src/DiatomGrowth/profit/template')
run_dir = os.path.expanduser('~/run/algae/vecma21')
exe = os.path.expanduser('~/src/DiatomGrowth/main.x')

with open(os.path.join(template_dir, 'input.txt'), 'r') as f:
    template = f.read()

np.random.seed(42)

nsamp0 = 64

keys = ['K_light', 'mu_0', 'f_Si', 'lambda_S', 'a', 'sigma_0']
mean = np.array([41.9, 1.19, 0.168, 0.0118, 1.25, 0.15])
stdev = 0.2*mean

def box_to_actual(x):
    return mean + stdev*np.sqrt(2.0)*erfinv(2.0*x - 1.0)

def actual_to_box(r):
    return 0.5*(1.0 + erf((r - mean)/(np.sqrt(2.0)*stdev)))

nvar = 6

r = halton(nsamp0, nvar)
rn = box_to_actual(r)

#%%
file_meas = os.path.join(
    run_dir, 'INPUT_DATA', 'GEESTHACHT', 'Chla_Fluor_2001.txt')
data_meas = pd.read_csv(
    file_meas, delim_whitespace=True, header=0, skiprows=[1],
    na_values=['empty'], parse_dates=[0], dayfirst=True)
data_meas = data_meas.groupby('Date').mean().interpolate(limit_direction='both')
data_meas = data_meas[data_meas.index >= '2001-03-12']
data_meas = data_meas[data_meas.index <= '2001-10-29']

fac_norm = 1.0/200.0  # To get values of O(1)
fac_meas = 5.2  # Conversion factor from Chla_Fluor to Chlorphyl concentration
yref = fac_meas*fac_norm*data_meas['Chla_Fluor'].values
nt = len(yref)
plt.figure()
plt.plot(yref)
#%%

def blackbox(x):
    params = {'year': 2001}
    for k, key in enumerate(keys):
        params[key] = x[k]

    input = profit.pre.replace_template(template, params)

    with open(os.path.join(run_dir, 'input.txt'), 'w') as f:
        f.write(input)

    run(exe, shell=True, text=True, cwd=run_dir)

    return fac_norm*np.loadtxt(
        os.path.join(run_dir, 'RESULTS', 'results.txt'), skiprows=1)

ymean = blackbox(mean)

#%%
plt.figure()
plt.plot(yref)
plt.plot(ymean)
#%%
def residuals(x):
    return yref - blackbox(x)

def cost(x):
    return np.sum(residuals(x)**2)/nt

#%%
rstart = mean

xopt, cov_x, infodict, mesg, ier = leastsq(
    residuals, x0=rstart, full_output=True)

print(xopt)

#%%
plt.figure()
plt.plot(yref)
plt.plot(blackbox(rstart))
plt.plot(blackbox(xopt), '--')
plt.legend(['reference', 'start', 'optimized'])


# %%
import GPy
from GPy.models import GPRegression
from warnings import catch_warnings
from warnings import simplefilter

k = GPy.kern.Matern52(nvar, ARD=True, lengthscale=0.2, variance=1)
mf = GPy.mappings.Linear(nvar, 1)

X = r.copy()
y = np.array([cost(box_to_actual(xk)) for xk in X])

model = GPRegression(X, y.reshape(-1,1), k,
        noise_var=1e-4)#, mean_function=mf)
model.optimize('bfgs')
print(model)
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
    Xsamples = np.random.rand(1024, nvar)
    scores = acquisition(Xsamples, model)
    ix = np.argmin(scores)
    return Xsamples[ix, :]

for i in range(nsamp0):
    x = opt_acquisition(X, model)
    ytrue = cost(box_to_actual(x))
    yest, _ = surrogate(model, x.reshape(-1, nvar))
    print(x, yest, ytrue)
    # add the data to the dataset
    X = np.vstack((X, x))
    y = np.append(y, ytrue)
    ymin = np.append(ymin, np.min(y))
    # update the model
    model.set_XY(X, y.reshape(-1, 1))
    model.optimize('bfgs')

# %%
plt.figure()
for k in range(len(X)):
    xt = box_to_actual(X[k, :])
    ypred, yvar = model.predict(actual_to_box(xt).reshape(-1, nvar))
    # plt.semilogy(k, cost(xt), 'rx')
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
kopt = np.argmin(y)
xopt = X[kopt,:]

plt.figure()
plt.plot(yref)
plt.plot(blackbox(mean), '-')
plt.plot(blackbox(box_to_actual(xopt)), '--')
plt.legend(['reference', 'start', 'optimized'])

# %%
plt.figure()
plt.semilogy(ymin)

# %% MCMC

sig2meas = (0.05*fac_norm)**2  # Measurement variance

nwarm = 50
nmc = 500

nstep = nwarm + nmc

# Input values and step sizes
x = np.empty((nstep + 1, nvar))
x[0, :] = box_to_actual(xopt)
sigprior = np.sqrt(sig2meas)*x[0, :]
dx = np.random.randn(nstep, nvar)*sigprior
xguess = np.empty(nvar)

acc = np.zeros((nstep + 1, nvar), dtype=bool)  # Acceptance rates
r = np.random.rand(nstep, nvar)  # Pre-computed random numbers

# Warmup
for k in range(nwarm):
    x[k+1, :] = x[k, :]
    pold = np.exp(-cost(x[k,:])/(2.0*sig2meas))  # likelihood
    for i in range(nvar):
        xguess[:] = x[k+1, :]
        xguess[i] += dx[k,i]
        pnew = np.exp(-cost(xguess)/(2.0*sig2meas))
        A = pnew/pold
        if A >= r[k,i]:
            x[k+1, :] = xguess
            pold = pnew
            acc[k,i] = True

plt.figure()
plt.plot(x[:nwarm, 0], x[:nwarm, 1])
plt.title('Warmup')

plt.figure()
plt.plot([np.exp(-cost(x[k,:])/(2.0*sig2meas)) for k in range(nwarm)])
plt.title('Warmup')

acceptance_rate = np.sum(acc[:nwarm], 0)/nwarm
print('Warmup acceptance rate: ', acceptance_rate)

#%%
target_rate = 0.3
dx = dx*np.exp(acceptance_rate/target_rate-1.0)

for k in range(nwarm, nstep):
    x[k+1, :] = x[k, :]
    pold = np.exp(-cost(x[k,:])/(2.0*sig2meas))  # likelihood
    for i in range(nvar):
        xguess[:] = x[k+1, :]
        xguess[i] += dx[k,i]
        pnew = np.exp(-cost(xguess)/(2.0*sig2meas))
        A = pnew/pold
        if A >= r[k, i]:
            x[k+1, :] = xguess
            pold = pnew
            acc[k,i] = True


plt.figure()
plt.plot(x[:, 0], x[:, 1])
plt.plot(x[:nwarm, 0], x[:nwarm, 1])
plt.title(f'MC, acceptance rate: {np.sum(acc[nwarm+1:], 0)/(nmc+1)}')

plt.figure()
plt.hist2d(x[nwarm+1:, 0], x[nwarm+1:, 1])
plt.figure()
plt.hist(x[nwarm+1:, 0])
plt.figure()
plt.hist(x[nwarm+1:, 1])

xmean = np.mean(x[nwarm+1:,:], 0)

print('Mean: ', xmean)
print('Variance: ', x[nwarm+1:].var(ddof=1))  # Unbiased variance
plt.figure()
pd.plotting.autocorrelation_plot(x[nwarm+1:nwarm+1000, 0])
pd.plotting.autocorrelation_plot(x[nwarm+1:nwarm+1000, 1])

# %% Delayed acceptance MCMC

# First testing surrogate
def cost_surrogate(x):
    return model.predict(actual_to_box(x).reshape(-1,2))[0][0,0]



# Input values and step sizes
x = np.empty((nstep + 1, nvar))
x[0, :] = box_to_actual(xopt)
sigprior = np.sqrt(sig2meas)*x[0, :]
dx = np.random.randn(nstep, nvar)*sigprior
xguess = np.empty(nvar)

acc1 = np.zeros((nstep + 1, nvar), dtype=bool)  # Acceptance rates sur
r1 = np.random.rand(nstep, nvar)  # Pre-computed random numbers for sur
acc2 = np.zeros((nstep + 1, nvar), dtype=bool)  # Acceptance rates true
r2 = np.random.rand(nstep, nvar)  # Pre-computed random numbers for true

# Warmup
for k in range(nwarm):
    x[k+1, :] = x[k, :]
    pold_sur = np.exp(-cost_surrogate(x[k, :])/(2.0*sig2meas))
    pold = np.exp(-cost(x[k, :])/(2.0*sig2meas))
    for i in range(nvar):
        xguess[:] = x[k+1, :]
        xguess[i] += dx[k, i]
        pnew_sur = np.exp(-cost_surrogate(xguess)/(2.0*sig2meas))
        A_sur = pnew_sur/pold_sur
        if A_sur >= r1[k, i]:
            acc1[k, i] = True
        else:  # Reject according to surrogate
            continue

        pnew = np.exp(-cost(xguess)/(2.0*sig2meas))
        A = (pnew/pold)/A_sur
        if A >= r2[k, i]:
            x[k+1, :] = xguess
            pold = pnew
            pold_sur = pnew_sur
            acc2[k, i] = True

plt.figure()
plt.plot(x[:nwarm, 0], x[:nwarm, 1])
plt.title('Warmup path')

plt.figure()
plt.plot([np.exp(-cost_surrogate(x[k, :])/(2.0*sig2meas)) for k in range(nwarm)])
plt.title('Warmup likelihood')

acceptance_rate = np.sum(acc2[:nwarm], 0)/nwarm
print('Warmup acceptance rate: ', acceptance_rate)

dx = dx*np.exp(acceptance_rate/target_rate-1.0)


for k in range(nwarm, nstep):
    x[k+1, :] = x[k, :]
    pold_sur = np.exp(-cost_surrogate(x[k, :])/(2.0*sig2meas))
    pold = np.exp(-cost(x[k, :])/(2.0*sig2meas))
    for i in range(nvar):
        xguess[:] = x[k+1, :]
        xguess[i] += dx[k, i]
        pnew_sur = np.exp(-cost_surrogate(xguess)/(2.0*sig2meas))
        A_sur = pnew_sur/pold_sur
        if A_sur >= r1[k, i]:
            acc1[k, i] = True
        else:  # Reject according to surrogate
            continue

        pnew = np.exp(-cost(xguess)/(2.0*sig2meas))
        A = (pnew/pold)/A_sur
        if A >= r2[k, i]:
            x[k+1, :] = xguess
            pold = pnew
            pold_sur = pnew_sur
            acc2[k, i] = True


plt.figure()
plt.plot(x[:, 0], x[:, 1])
plt.plot(x[:nwarm, 0], x[:nwarm, 1])
plt.title(f'MC, acceptance rates: \
    {np.sum(acc1[nwarm+1:], 0)/(nmc+1),np.sum(acc2[nwarm+1:], 0)/(nmc+1)}')

plt.figure()
plt.hist2d(x[nwarm+1:, 0], x[nwarm+1:, 1])
plt.figure()
plt.hist(x[nwarm+1:, 0])
plt.figure()
plt.hist(x[nwarm+1:, 1])

print('Mean: ', xmean)
print('Variance: ', x[nwarm+1:].var(ddof=1))  # Unbiased variance
plt.figure()
pd.plotting.autocorrelation_plot(x[nwarm+1:nwarm+1000, 0])
pd.plotting.autocorrelation_plot(x[nwarm+1:nwarm+1000, 1])

# %%
