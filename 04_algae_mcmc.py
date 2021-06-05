# %%
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from multiprocessing import Process
from algae_common import *

nproc = 16

niwarm = 2
nwarm = 250
nmc = 1000

def main():
    workers = []
    for kproc in range(nproc):
        run_dir = os.path.join(base_dir, f'{kproc}')
        init_dir(run_dir)

        p = Process(target=start_run,
            args=(os.path.join(base_dir, f'{kproc}'), kproc,))
        workers.append(p)
        p.start()

    for kproc, worker in enumerate(workers):
        worker.join()

def start_run(run_dir, kproc):
    np.random.seed(kproc)
    tic = time()
    # MCMC
    sig_ch = 5.0  # Basic uncertainty
    sig2meas = (sig_ch*fac_norm)**2  # Measurement variance

    nstep = nwarm + nmc

    # Input values and step sizes
    x = np.empty((nstep + 1, nvar))
    x[0, :] = np.load('ropt.npy')[:nvar]
    sigprior = np.sqrt(sig2meas)*x[0, :]
    dx = np.random.randn(nstep, nvar)*sigprior
    xguess = np.empty(nvar)

    # Warmup
    for ki in range(niwarm):
        acc = np.zeros((nstep + 1, nvar), dtype=bool)  # Acceptance rates
        r = np.log(np.random.rand(nstep, nvar))  # Pre-computed random numbers
        lpold = -cost(x[0, :], run_dir)/(2.0*sig2meas) # log-likelihood
        for k in range(nwarm):
            x[k+1, :] = x[k, :]
            for i in range(nvar):
                xguess[:] = x[k+1, :]
                xguess[i] += dx[k, i]
                xguess[i] = np.abs(xguess[i])  # Mirror negative values
                lpnew = -cost(xguess, run_dir)/(2.0*sig2meas)
                A = lpnew - lpold
                if A >= r[k, i]:
                    x[k+1, :] = xguess
                    lpold = lpnew
                    acc[k,i] = True

        target_rate = 0.35
        acceptance_rate = np.sum(acc[:nwarm], 0)/nwarm
        dx = dx*np.exp(acceptance_rate/target_rate-1.0)
        if ki < niwarm:
            x[0, :] = x[nwarm, :]

    plt.figure()
    plt.plot(x[:nwarm, 0], x[:nwarm, 1])
    plt.title(f'warmup, acceptance rate: {np.sum(acc[:nwarm], 0)/(nwarm+1)}')
    plt.savefig(os.path.join(run_dir, '1.png'))

    print('Warmup acceptance rate: ', acceptance_rate)

    lpold = -cost(x[nwarm, :], run_dir)/(2.0*sig2meas)  # Log-likelihood
    for k in range(nwarm, nstep):
        x[k+1, :] = x[k, :]
        for i in range(nvar):
            xguess[:] = x[k+1, :]
            xguess[i] += dx[k,i]
            xguess[i] = np.abs(xguess[i])  # Mirror negative values
            lpnew = -cost(xguess, run_dir)/(2.0*sig2meas)
            A = lpnew - lpold
            if A >= r[k, i]:
                x[k+1, :] = xguess
                lpold = lpnew
                acc[k,i] = True
    toc = time() - tic

    plt.figure()
    plt.plot(x[:, 0], x[:, 1])
    plt.plot(x[:nwarm, 0], x[:nwarm, 1])
    plt.title(f'MC, acceptance rate: {np.sum(acc[nwarm+1:], 0)/(nmc+1)}')
    plt.savefig(os.path.join(run_dir, '2.png'))

    plt.figure()
    plt.hist2d(x[nwarm+1:, 0], x[nwarm+1:, 1])
    plt.figure()
    plt.hist(x[nwarm+1:, 0])
    plt.figure()
    plt.hist(x[nwarm+1:, 1])
    plt.savefig(os.path.join(run_dir, '3.png'))

    print('Mean: ', np.mean(x[nwarm+1:,:], axis = 0))
    print('Variance: ', x[nwarm+1:].var(axis=0, ddof=1))  # Unbiased variance
    plt.figure()
    pd.plotting.autocorrelation_plot(x[nwarm+1:, 0])
    pd.plotting.autocorrelation_plot(x[nwarm+1:, 1])
    plt.savefig(os.path.join(run_dir, '4.png'))

    np.savetxt(
        os.path.join(run_dir, f'runtime.txt'), np.array([toc]), fmt='%.2e')
    np.save(f'xmc_{kproc}', x)

if __name__ == '__main__':
    main()
