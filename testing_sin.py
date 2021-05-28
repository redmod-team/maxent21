# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import profit
from scipy.optimize import leastsq
from scipy.special import erfinv, erf
from scipy.stats import norm
from profit.util.halton import halton

np.random.seed(42)

nsamp0 = 64

mean = np.array([1.0, 1.0])
stdev = np.array([0.2, 0.3])

def box_to_actual(x):
    return mean + stdev*np.sqrt(2.0)*erfinv(2.0*x - 1.0)

def actual_to_box(r):
    return 0.5*(1.0 + erf((r - mean)/(np.sqrt(2.0)*stdev)))

r = halton(nsamp0, 2)
rn = box_to_actual(r)

#%%
nt = 250
t = np.linspace(0, 2.0*(1.0-1.0/nt), nt)

xpath = []
def blackbox(x):
    xpath.append(x)
    return x[0]*(t - x[1])**3

# Reference values for optimum
xref = np.array([1.15, 1.4])
yref = blackbox(xref)

def residuals(x):
    return yref - blackbox(x)

def cost(x):
    return np.sum(residuals(x)**2)/nt

# %%
rstart = mean

xpath = []
xopt, cov_x, infodict, mesg, ier = leastsq(
    residuals, x0=rstart, full_output=True)
xpath0 = np.array(xpath.copy())

print(xopt)

plt.figure()
plt.plot(xpath0[:,0], xpath0[:,1])

plt.figure()
plt.plot(yref)
plt.plot(blackbox(rstart))
plt.plot(blackbox(xopt), '--')
plt.legend(['reference', 'start', 'optimized'])


# %%
import GPy
from GPy.models import GPRegression
from scipy.optimize import minimize
from warnings import catch_warnings
from warnings import simplefilter

k = GPy.kern.Matern52(2, ARD=True, lengthscale=0.1, variance=1)
mf = GPy.mappings.Linear(2, 1)

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
    mu, std = surrogate(model, x.reshape(-1,2))
    std = std + 1e-31
    probs = (ymin[-1] - mu)*norm.cdf(ymin[-1], mu, std) + \
        std**2 * norm.pdf(ymin[-1], mu, std)
    return -probs

def opt_acquisition(X, model):
    Xsamples = np.random.rand(1024, 2)
    scores = acquisition(Xsamples, model)
    ix = np.argmin(scores)
    return Xsamples[ix, :]

# def opt_acquisition(X, model):
#     x0 = opt_acquisition_rand(X, model)
#     res = minimize(acquisition, x0, args=model, bounds=[(0,0.9), (0,0.9)])
#     print(res.success)
#     if res.success and np.all(x > 0)  and np.all(x < 1):
#         return res.x
#     return x0

for i in range(nsamp0):
    x = opt_acquisition(X, model)
    ytrue = cost(box_to_actual(x))
    yest, _ = surrogate(model, x.reshape(-1, 2))
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
plt.plot(blackbox(box_to_actual(xopt)), '--')
plt.legend(['reference', 'start', 'optimized'])

# %%
plt.figure()
plt.semilogy(ymin)

# # %% MCMC

sig2meas = 0.02**2  # Measurement variance

nwarm = 500
nmc = 10000

nvar = 2

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

target_rate = 0.35
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
plt.plot(xref[0], xref[1], 'rx')
plt.figure()
plt.hist(x[nwarm+1:, 0])
plt.plot(xref[0], 0, 'rx')
plt.figure()
plt.hist(x[nwarm+1:, 1])
plt.plot(xref[1], 0, 'rx')

xmean = np.mean(x[nwarm+1:,:], 0)
c0 = np.correlate(x[nwarm+1:,0]-xmean[0], x[nwarm+1:,0]-xmean[0], mode='full')
c1 = np.correlate(x[nwarm+1:,0]-xmean[1], x[nwarm+1:,1]-xmean[1], mode='full')
plt.figure()
plt.plot(c0[len(c0)//2:][:1000])
plt.plot(c1[len(c1)//2:][:1000])
plt.title('Autocorrelation')

print('Mean: ', xmean)
print('Variance: ', x[nwarm+1:].var(ddof=1))  # Unbiased variance

# %% Delayed acceptance MCMC

# First testing surrogate
def cost_surrogate(x):
    return model.predict(actual_to_box(x).reshape(-1,2))[0][0,0]

plt.figure()
for k in range(len(X)):
    xt = box_to_actual(X[k, :])
    ypred, yvar = model.predict(actual_to_box(xt).reshape(-1,2))
    plt.semilogy(k, cost(xt), 'rx')
    plt.semilogy([k, k], [ypred[0,0], ypred[0,0] + 2*np.sqrt(yvar[0,0])], 'k-')

sig2meas = 0.02**2  # Measurement variance

nwarm = 500
nmc = 10000

nvar = 2

nstep = nwarm + nmc

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

target_rate = 0.35
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
plt.plot(xref[0], xref[1], 'rx')
plt.figure()
plt.hist(x[nwarm+1:, 0])
plt.plot(xref[0], 0, 'rx')
plt.figure()
plt.hist(x[nwarm+1:, 1])
plt.plot(xref[1], 0, 'rx')

xmean = np.mean(x[nwarm+1:,:], 0)
c0 = np.correlate(x[nwarm+1:,0]-xmean[0], x[nwarm+1:,0]-xmean[0], mode='full')
c1 = np.correlate(x[nwarm+1:,0]-xmean[1], x[nwarm+1:,1]-xmean[1], mode='full')
plt.figure()
plt.plot(c0[len(c0)//2:][:1000])
plt.plot(c1[len(c1)//2:][:1000])
plt.title('Autocorrelation')

print('Mean: ', xmean)
print('Variance: ', x[nwarm+1:].var(ddof=1))  # Unbiased variance

# %%
