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

nsamp0 = 16

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
from sklearn.gaussian_process import GaussianProcessRegressor
import GPy
from scipy.optimize import minimize
from warnings import catch_warnings
from warnings import simplefilter

model = GaussianProcessRegressor()
# k = GPy.kern.Matern52(2, ARD=True, lengthscale=0.2, variance=0.1**2)
# mf = GPy.mappings.Linear(2, 1)


X = r.copy()
y = np.array([cost(box_to_actual(xk)) for xk in X])

# model = GPy.models.GPRegression(X, y.reshape(-1,1), k,
#         noise_var=1e-4, mean_function=mf)

#%%

model.fit(X, y)

ymin = np.array([np.min(y)])

def surrogate(model, X):
	with catch_warnings():
		simplefilter("ignore")
		return model.predict(X, return_std=True)

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
# 	x0 = opt_acquisition_rand(X, model)
# 	res = minimize(acquisition, x0, args=model, bounds=[(0,0.9), (0,0.9)])
# 	print(res.success)
# 	if res.success and np.all(x > 0)  and np.all(x < 1):
# 		return res.x
# 	return x0

for i in range(8):
	x = opt_acquisition(X, model)
	ytrue = cost(box_to_actual(x))
	yest, _ = surrogate(model, x.reshape(-1, 2))
	print(x, yest, ytrue)
	# add the data to the dataset
	X = np.vstack((X, x))
	y = np.append(y, ytrue)
	ymin = np.append(ymin, np.min(y))
	# update the model
	model.fit(X, y)
# %%
plt.figure()
plt.scatter(X[:,0], X[:,1], c=model.predict(X))
plt.colorbar()

plt.figure()
XN = box_to_actual(X)
plt.scatter(XN[:,0], XN[:,1], c=model.predict(X))
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

# %% MCMC

sig2meas = 0.02**2  # Measurement variance

def likelihood(x):
    return np.exp(-cost(x)/(2.0*sig2meas))


def step(dim, x, dx, r, p):
    xguess = x.copy()
    xguess[dim] += dx[dim]
    A = p(xguess)/p(x)
    if A >= 1:
        return xguess, True
    if A >= r[dim]:
        return xguess, True
    return x, False


nwarm = 500
nmc = 10000

nvar = 2

nstep = nwarm + nmc

# Input values and step sizes
x = np.empty((nstep + 1, nvar))
x[0, :] = box_to_actual(xopt)
sigprior = np.sqrt(sig2meas)*x[0, :]
dx = np.random.randn(nstep, nvar)*sigprior

acc = np.empty((nstep + 1, nvar), dtype=bool)  # Acceptance rates
r = np.random.rand(nstep, nvar)  # Pre-computed random numbers

# Warmup
for k in range(nwarm):
    x[k+1, :] = x[k, :]
    for i in range(nvar):
        x[k+1, :], acc[k, i] = step(i, x[k+1, :], dx[k, :], r[k, :], likelihood)

plt.figure()
plt.plot(x[:nwarm, 0], x[:nwarm, 1])
plt.title('Warmup')

plt.figure()
plt.plot([likelihood(x[k, :]) for k in range(nwarm)])
plt.title('Warmup')

acceptance_rate = np.sum(acc[:nwarm], 0)/nwarm
print('Warmup acceptance rate: ', acceptance_rate)

target_rate = 0.35
dx = dx*np.exp(acceptance_rate/target_rate-1.0)

for k in range(nwarm, nstep):
    x[k+1, :] = x[k, :]
    for i in range(nvar):
        x[k+1, :], acc[k, i] = step(i, x[k+1, :], dx[k, :], r[k, :], likelihood)


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

#%%
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
def cost_surrogate(x):
    return model.predict(actual_to_box(x).reshape(-1,2))[0]

xt = box_to_actual(xopt)
print(model.predict(actual_to_box(xt).reshape(-1,2), return_std=True), cost(xt))

# %%
