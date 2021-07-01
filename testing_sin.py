# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import leastsq
from scipy.stats import norm
from profit.util.halton import halton
from mcmc_common import *

np.random.seed(42)

nsamp0 = 32

sig2meas = 0.05**2  # Measurement variance

mean = np.array([1.0, 1.0])

# Uniform distribution
length = np.array([2.0, 2.0])

def box_to_actual(x):
    return mean + (x - 0.5)*length

def actual_to_box(r):
    return (r - mean)/length + 0.5

# Normal distribution
# stdev = np.array([0.2, 0.3])
#
# def box_to_actual(x):
#     return mean + stdev*np.sqrt(2.0)*erfinv(2.0*x - 1.0)
#
# def actual_to_box(r):
#     return 0.5*(1.0 + erf((r - mean)/(np.sqrt(2.0)*stdev)))

# Cauchy distribution
# thstar = np.array([0.2, 0.2])
# Pstar = 0.9
# b = thstar*np.arctan(np.pi/2.0*Pstar)
# def box_to_actual(x):
#     return mean + b*np.tan(np.pi*(x - 0.5))

# def actual_to_box(r):
#     return 1.0/np.pi*(np.arctan((r - mean)/b)) + 0.5

# Sample only in circle
r = halton(1024, 2)
# r = r[np.sum((r - 0.5)**2, 1) < 0.25, :]
r = r[:nsamp0, :]

rn = box_to_actual(r)

plt.figure()
plt.plot(rn[:,0], rn[:,1], 'x')
plt.axis('equal')
#%%
nt = 250
t = np.linspace(0, 2.0*(1.0-1.0/nt), nt)

xpath = []
def blackbox(x):
    xpath.append(x)
    ret = x[0]*np.sin((t - x[1])**3)
    #ret[ret<-10] = -10
    #ret[ret>10] = 10
    return ret

# Reference values for optimum
xref = np.array([1.15, 1.4])
yref = blackbox(xref)
yref = yref + np.sqrt(sig2meas)*np.random.randn(len(yref))

def residuals(x):
    return yref - blackbox(x)

def cost(x):
    return np.sum(residuals(x)**2)/(nt*2.0*sig2meas)

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

kernel = GPy.kern.Matern52(2, ARD=True, lengthscale=2.0/nsamp0, variance=1)
mf = GPy.mappings.Linear(2, 1)

X = r.copy()
y = np.array([cost(box_to_actual(xk))*2.0*sig2meas for xk in X])

model = GPRegression(X, y.reshape(-1,1), kernel,
        noise_var=1e-4, mean_function=mf)
model.optimize('bfgs')
print(model)
# %% Bayesian optimization

ymin = np.array([np.min(y)])

def surrogate(X):
    with catch_warnings():
        simplefilter("ignore")
        mu, var = model.predict(X, full_cov=False)
        return mu/(2.0*sig2meas), np.sqrt(var)/(2.0*sig2meas)

def acquisition(x):
    mu, std = surrogate(x.reshape(-1, 2))
    std = std + 1e-31
    probs = (ymin[-1] - mu)*norm.cdf(ymin[-1], mu, std) + \
        std**2 * norm.pdf(ymin[-1], mu, std)
    return -probs

def opt_acquisition():
    Xsamples = np.random.rand(1024, 2)
    scores = acquisition(Xsamples)
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
    x = opt_acquisition()
    ytrue = cost(box_to_actual(x))*(2.0*sig2meas)
    yest, _ = surrogate(x.reshape(-1, 2))
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

plt.figure()
for k in range(len(X)):
    xt = box_to_actual(X[k, :])
    plt.semilogy(k, cost(xt), 'rx')

for k in range(len(X)):
    xt = box_to_actual(X[k, :])
    #ypred, yvar = model.predict(actual_to_box(xt).reshape(-1,2))
    ypred, yvar = surrogate(actual_to_box(xt).reshape(-1,2))
    plt.semilogy([k, k],
        [ypred[0,0],
         ypred[0,0] + max(2*np.sqrt(yvar[0,0]), 0.1*ypred[0,0])], 'k-')


plt.figure()
plt.semilogy(ymin)

# %% MCMC

nvar = 2
niwarm = 5
nwarm = 500
nmc = 10000
nstep = nwarm + nmc

dx = np.random.randn(nstep, nvar)*np.sqrt(sig2meas)*box_to_actual(xopt)
x, acc = mcmc(box_to_actual(xopt), dx, niwarm, nwarm, nmc, cost)

plt.figure()
plt.plot(x[:nwarm, 0], x[:nwarm, 1], ',')
plt.title('Warmup')

plt.figure()
plt.plot([np.exp(-cost(x[k,:])) for k in range(nwarm)])
plt.title('Warmup')

bins = [np.linspace(0.0, 2.0, 21), np.linspace(0, 1.8, 21)]

def plot_mc_results(name, acor=10, figsize=[4.0, 4.0], figpath = 'paper/fig'):
    figs = []

    figs.append(plt.figure(figsize=[4.0, 2.0]))
    pd.plotting.autocorrelation_plot(x[nwarm+1:, 0])
    pd.plotting.autocorrelation_plot(x[nwarm+1:, 1], linestyle='dashed')
    plt.xlim(0, 1000)
    plt.tight_layout()
    plt.savefig(os.path.join(figpath, f'acor_{name}.pdf'))


    figs.append(plt.figure())
    sns.displot(
        x = x[nwarm+1::acor, 0], y = x[nwarm+1::acor, 1], height=3.0,
        bins=bins, aspect=4.0/3.0)
    plt.plot(x[nwarm+1::acor, 0], x[nwarm+1::acor, 1], 'k,', alpha=1.0)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xlim([0, 2])
    plt.ylim([1, 1.8])
    plt.tight_layout()
    plt.savefig(os.path.join(figpath, f'hist2_{name}.pdf'))

    figs.append(plt.figure())
    sns.displot(x[nwarm+1::acor, 0], kde=True, height=figsize[0])
    plt.xlabel('$x_1$')
    plt.xlim([0, 2])
    plt.tight_layout()
    plt.savefig(os.path.join(figpath, f'hist_x1_{name}.pdf'))

    figs.append(plt.figure())
    sns.displot(x[nwarm+1::acor, 1], kde=True, height=figsize[0])
    plt.xlabel('$x_2$')
    plt.xlim([1, 1.8])
    plt.tight_layout()
    plt.savefig(os.path.join(figpath, f'hist_x2_{name}.pdf'))

    return figs


figs = plot_mc_results('mcmc')



# %% MCMC hierachical
from scipy.special import gamma

def cost_hi(x):
    prior = (x[-1]+1.0 - 2.0)**2 / (2.0*0.5**2)
    lik = np.sum(
        np.abs(residuals(x[:-1])/np.sqrt(2.0*sig2meas))**(x[-1]+1.0))/nt
    lik = lik + np.log(gamma(1.0 + 1.0/(x[-1]+1.0)))
    return lik + prior

x0 = np.empty(nvar+1)
dx = np.empty((nstep, nvar+1))

x0[:-1] = box_to_actual(xopt)
x0[-1] = 2.0 - 1.0  # L2 norm
dx[:,:-1] = np.random.randn(nstep, nvar)*np.sqrt(sig2meas)*box_to_actual(xopt)
dx[:,-1] = np.random.randn(nstep)*0.1
x, acc = mcmc(x0, dx, niwarm, nwarm, nmc, cost_hi)

plt.figure()
plt.plot(x[:nwarm, 0], x[:nwarm, 1])
plt.title('Warmup')

plt.figure()
plt.plot([np.exp(-cost(x[k,:])) for k in range(nwarm)])
plt.title('Warmup')

print(f'MC, acceptance rate: {np.sum(acc[nwarm+1:], 0)/(nmc+1)}')

figs = plot_mc_results('mcmc_hi')


figs.append(plt.figure(figsize=(4.0,3.0)))
sns.displot(x[nwarm+1::10, 2] + 1, height=2.0, aspect=2.0, kde=True)
plt.xlabel(r'$\theta$')
plt.xlim([1, 3.5])
plt.tight_layout()
plt.savefig('paper/fig/hist_theta_mcmc_hi.pdf')

xmean = np.mean(x[nwarm+1:,:], 0)
print('Mean: ', xmean)
print('Variance: ', x[nwarm+1:].var(ddof=1))  # Unbiased variance
# %% Delayed acceptance MCMC

# def cost_surrogate(x):
#     return surrogate(actual_to_box(x).reshape(-1,2))[0]

# # Input values and step sizes
# x0 = box_to_actual(xopt)
# dx = np.random.randn(nstep, nvar)*np.sqrt(sig2meas)*x0

# x, acc1, acc2 = mcmc(x0, dx, niwarm, nwarm, nmc, cost, cost_surrogate)

# plt.figure()
# plt.plot(x[:nwarm, 0], x[:nwarm, 1])
# plt.title('Warmup path')

# plt.figure()
# plt.plot([np.exp(-cost_surrogate(x[k, :])[0]) for k in range(nwarm)])
# plt.title('Warmup likelihood')

# # plot_mc_results()

# %%
from profit.sur.linear_reduction import KarhunenLoeve

X = r.copy()
y = np.array([blackbox(box_to_actual(xk)) for xk in X])

# %%
kl = KarhunenLoeve(y, tol=1e1)

fig, ax = plt.subplots(figsize=(4.0, 3.0))
ax.loglog(1, kl.w[-1]/kl.w[-1], 'x')
for k in range(kl.w.shape[0]):
    ax.loglog(k+1, kl.w[-k-1]/kl.w[-1], 'x')
ax.set_xlabel('Index')
ax.set_ylabel('Eigenvalues')
fig.tight_layout()

ztrain = kl.project(y)

fig, ax = plt.subplots(figsize=(4.0, 3.0))
ax.plot(kl.ymean)
ax.plot(-kl.features()[:,::-1])
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$g(\tau)$')
ax.legend(['mean'] + [f'$\\varphi_{k+1}$' for k in range(3)], loc='upper right')
fig.tight_layout()

# %%

z = kl.project(y)

models = []
for zk in z:
    kernel = GPy.kern.Matern52(2, ARD=True, lengthscale=2.0/nsamp0, variance=1)
    mf = GPy.mappings.Linear(2, 1)
    model = GPRegression(X, zk.reshape(-1, 1), kernel,
        noise_var=1e-4, mean_function=mf)
    model.optimize('bfgs')
    print(model)
    models.append(model)

# %%
neig = len(kl.w)
mus = np.empty((neig, nsamp0))
vars = np.empty((neig, nsamp0))
for k, model in enumerate(models):
    mu, var = model.predict(X*1.01, full_cov=False)
    mus[k, :] = mu.flat
    vars[k, :] = var.flat

# %% Delayed acceptance MCMC II

def residuals_y(y):
    return yref - y


def cost_y(y):
    return np.sum(residuals_y(y)**2, 1)/(nt*2.0*sig2meas)


def surrogate_kl(X):
    with catch_warnings():
        simplefilter("ignore")
        mus = np.empty((neig, X.shape[0]))
        for k, model in enumerate(models):
            mu, _ = model.predict(X, full_cov=False)
            mus[k, :] = mu.flat

        ymu, yvars = kl.lift(mus, vars)
        ycost = cost_y(ymu)

    return ycost, 0.0  # TODO: variance


def cost_surrogate_kl(x):
    return surrogate_kl(actual_to_box(x).reshape(-1,2))[0]


# Input values and step sizes
x0 = box_to_actual(xopt)
dx = np.random.randn(nstep, nvar)*np.sqrt(sig2meas)*x0

x, acc1, acc2 = mcmc(x0, dx, niwarm, nwarm, nmc, cost, cost_surrogate_kl)

plt.figure()
plt.plot(x[:nwarm, 0], x[:nwarm, 1], ',')
plt.title('Warmup path')

plt.figure()
plt.plot([np.exp(-cost_surrogate_kl(x[k, :])[0]) for k in range(nwarm)])
plt.title('Warmup likelihood')

figs = plot_mc_results('mcmc_kl_da')

# %% Delayed acceptance MCMC hierachical

def residuals_y(y):
    return yref - y


def cost_y_hi(y, th):
    prior = (th+1.0 - 2.0)**2 / (2.0*0.5**2)
    lik = np.sum(
        np.abs(residuals_y(y)/np.sqrt(2.0*sig2meas))**(th+1.0))/nt
    lik = lik + np.log(gamma(1.0 + 1.0/(th+1.0)))
    return lik + prior


def surrogate_kl_hi(X, th):
    with catch_warnings():
        simplefilter("ignore")
        mus = np.empty((neig, X.shape[0]))
        for k, model in enumerate(models):
            mu, _ = model.predict(X, full_cov=False)
            mus[k, :] = mu.flat

        ymu, yvars = kl.lift(mus, vars)
        ycost = cost_y_hi(ymu, th)

    return ycost, 0.0  # TODO: variance


def cost_surrogate_kl_hi(x):
    return surrogate_kl_hi(actual_to_box(x[:-1]).reshape(-1, 2), x[-1])[0]

x0 = np.empty(nvar+1)
dx = np.empty((nstep, nvar+1))

x0[:-1] = box_to_actual(xopt)
x0[-1] = 2.0 - 1.0  # L2 norm
dx[:,:-1] = np.random.randn(nstep, nvar)*np.sqrt(sig2meas)*box_to_actual(xopt)
dx[:,-1] = np.random.randn(nstep)*0.1


x, acc1, acc2 = mcmc(x0, dx, niwarm, nwarm, nmc, cost_hi, cost_surrogate_kl_hi)


plt.figure()
plt.plot(x[:nwarm, 0], x[:nwarm, 1], ',')
plt.title('Warmup path')

plt.figure()
plt.plot([np.exp(-cost_surrogate_kl(x[k, :-1])[0]) for k in range(nwarm)])
plt.title('Warmup likelihood')

figs = plot_mc_results('mcmc_kl_da_hi')

figs.append(plt.figure(figsize=(4.0,3.0)))
sns.displot(x[nwarm+1::10, 2] + 1, height=2.0, aspect=2.0, kde=True)
plt.xlabel(r'$\theta$')
plt.xlim([1, 3.5])
plt.tight_layout()
plt.savefig('paper/fig/hist_theta_mcmc_kl_da_hi.pdf')

#%% TODO: Bayesian optimization with KL surrogate

# ycost = cost_y(y)
# ymin = np.array([np.min(ycost)])
# kopt = np.argmin(ycost)
# xopt = X[kopt, :]

# def surrogate_kl(X):
#     with catch_warnings():
#         simplefilter("ignore")
#         mus = np.empty((neig, X.shape[0]))
#         vars = np.empty((neig, X.shape[0]))
#         for k, model in enumerate(models):
#             mu, var = model.predict(X, full_cov=False)
#             mus[k, :] = mu.flat
#             vars[k, :] = var.flat

#         ymu, yvars = kl.lift(mus, vars)
#         ycost = cost_y(ymu)
#         dycost2 = residuals_y(ymu).T**2/(ycost*(nt*2.0*sig2meas)**2)
#         ycostvar = np.sum(dycost2.T*yvars, 1)

#     return ycost, ycostvar

# def acquisition_kl(x):
#     mu, std = surrogate_kl(x.reshape(-1, 2))
#     std = std + 1e-31
#     probs = (ymin[-1] - mu)*norm.cdf(ymin[-1], mu, std) + \
#         std**2 * norm.pdf(ymin[-1], mu, std)
#     return -probs

# def opt_acquisition_kl():
#     Xsamples = np.random.rand(1024, 2)
#     scores = acquisition_kl(Xsamples)
#     ix = np.argmin(scores)
#     return Xsamples[ix, :]

# # %%

# for i in range(nsamp0):
#     x = opt_acquisition_kl()
#     ytrue = blackbox(box_to_actual(x))
#     ycost_true = cost_y(ytrue)*(2.0*sig2meas)
#     yest, _ = surrogate_kl(x.reshape(-1, 2))
#     print(x, yest, ytrue)
#     # add the data to the dataset
#     X = np.vstack((X, x))
#     ycost = np.append(y, ycost_true)
#     ymin = np.append(ymin, np.min(ycost))
#     # update the model
#     model.set_XY(X, y.reshape(-1, 1))
#     model.optimize('bfgs')


# # %%

# plt.figure()
# plt.plot(yref)
# plt.plot(blackbox(box_to_actual(xopt)), '--')
# plt.legend(['reference', 'start', 'optimized'])

# plt.figure()
# for k in range(len(X)):
#     xt = box_to_actual(X[k, :])
#     ypred, yvar = surrogate_kl(actual_to_box(xt).reshape(-1,2))
#     plt.semilogy(k, cost(xt), 'rx')
#     plt.semilogy([k, k],
#         [ypred,
#          ypred + max(2*np.sqrt(yvar), 0.1*ypred)], 'k-')


# plt.figure()
# plt.semilogy(ymin)

# # %%
