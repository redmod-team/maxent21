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

mean = np.array([1.0, 1.0, np.sqrt(sig2meas)])

nt = 25
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
    return np.sum(residuals(x)**2)/(2.0*sig2meas) + 0.5*nt*np.log(sig2meas)

def vary_sig(x):
    global sig2meas
    sig2meas = x[-1]**2

plt.figure()
plt.plot(yref)

# %% MCMC

nvar = len(mean)
niwarm = 5
nwarm = 500
nmc = 10000
nstep = nwarm + nmc

dx = np.random.randn(nstep, nvar)*np.sqrt(sig2meas)*mean
x, acc = mcmc(mean, dx, niwarm, nwarm, nmc, cost, vary_sig=vary_sig)

plt.figure()
plt.plot(x[:nwarm, 0], x[:nwarm, 1], ',')
plt.title('Warmup')

plt.figure()
plt.plot([np.exp(-cost(x[k,:])) for k in range(nwarm)])
plt.title('Warmup')


def plot_mc_results(name, acor=10, figsize=[4.0, 4.0], figpath = 'paper/fig'):
    figs = []

    figs.append(plt.figure(figsize=[4.0, 2.0]))
    pd.plotting.autocorrelation_plot(x[nwarm+1:, 0])
    pd.plotting.autocorrelation_plot(x[nwarm+1:, 1], linestyle='dashed')
    #plt.xlim(0, 1000)
    plt.tight_layout()
    plt.savefig(os.path.join(figpath, f'acor_{name}.pdf'))


    figs.append(plt.figure())
    sns.displot(
        x = x[nwarm+1::acor, 0], y = x[nwarm+1::acor, 1], height=3.0,
        aspect=4.0/3.0)
    plt.plot(x[nwarm+1::acor, 0], x[nwarm+1::acor, 1], 'k,', alpha=1.0)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    #plt.xlim([0, 2])
    #plt.ylim([1, 1.8])
    plt.tight_layout()
    plt.savefig(os.path.join(figpath, f'hist2_{name}.pdf'))

    figs.append(plt.figure())
    sns.displot(x[nwarm+1::acor, 0], kde=True, height=figsize[0])
    plt.xlabel('$x_1$')
    #plt.xlim([0, 2])
    plt.tight_layout()
    plt.savefig(os.path.join(figpath, f'hist_x1_{name}.pdf'))

    figs.append(plt.figure())
    sns.displot(x[nwarm+1::acor, 1], kde=True, height=figsize[0])
    plt.xlabel('$x_2$')
    #plt.xlim([1, 1.8])
    plt.tight_layout()
    plt.savefig(os.path.join(figpath, f'hist_x2_{name}.pdf'))
    
    figs.append(plt.figure())
    sns.displot(x[nwarm+1::acor, 2], kde=True, height=figsize[0])
    plt.xlabel('$sig$')
    #plt.xlim([1, 1.8])
    plt.tight_layout()
    plt.savefig(os.path.join(figpath, f'hist_sig_{name}.pdf'))

    return figs


figs = plot_mc_results('mcmc', figpath='/tmp')
