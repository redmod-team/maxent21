# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import profit
from scipy.optimize import leastsq
from scipy.special import erfinv
from subprocess import run
from profit.util.halton import halton

np.random.seed(42)

nsamp0 = 16

mean = np.array([1.0, 1.0])
stdev = np.array([0.2, 0.3])
def box_to_actual(x):
    return mean + stdev*np.sqrt(2.0)*erfinv(2.0*x - 1.0)

r = halton(nsamp0, 2)
rn = box_to_actual(r)

#%%
nt = 250
t = np.linspace(0, 2.0*np.pi*(1.0-1.0/nt), nt)

xpath = []
def blackbox(x):
    xpath.append(x)
    return x[0]*np.sin(x[1]*t)

# Reference values for optimum
xref = np.array([1.15, 1.4])
yref = blackbox(xref)

def residuals(x):
    return yref - blackbox(x)

def cost(x):
    return np.sqrt(np.sum(residuals(x)**2)/nt)

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
from scipy.stats import norm
from warnings import catch_warnings
from warnings import simplefilter

model = GaussianProcessRegressor()

X = r.copy()
y = np.array([cost(box_to_actual(xk)) for xk in X])
model.fit(X, y)


def surrogate(model, X):
	with catch_warnings():
		simplefilter("ignore")
		return model.predict(X, return_std=True)

def acquisition(X, Xsamples, model):
	yhat, _ = surrogate(model, X)
	best = np.min(yhat)
	mu, std = surrogate(model, Xsamples)
	probs = norm.cdf((mu - best) / (std+1E-9))
	return probs

def opt_acquisition(X, y, model):
	Xsamples = np.random.rand(1024, 2)
	scores = acquisition(X, Xsamples, model)
	ix = np.argmin(scores)
	return Xsamples[ix, :]

ymin = np.array([])
for i in range(20):
	x = opt_acquisition(X, y, model)
	ytrue = cost(box_to_actual(x))
	yest, _ = surrogate(model, x.reshape(-1, 2))
	print(x, yest, ytrue)
	# add the data to the dataset
	X = np.vstack((X, x))
	y = np.append(y, ytrue)
	ymin = np.append(ymin, min(y))
	# update the model
	model.fit(X, y)
# %%
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
plt.semilogy(ymin)

# %%
