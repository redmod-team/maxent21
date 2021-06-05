# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import GPy
import h5py
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
    # MCMC with delayed acceptance

    k = GPy.kern.Matern52(nvar, ARD=True, lengthscale=0.2, variance=1)
    mf = GPy.mappings.Linear(nvar, 1)

    X = np.load('Xtrain0.npy')
    y = cost_y(np.load('ytrain0.npy'))
    model = GPy.models.GPRegression(X, y.reshape(-1, 1), k,
            noise_var=1e-4, mean_function=mf)

    with h5py.File('sur0.hdf5', 'r') as f:
        model.param_array[:] = f['param_array']

    def cost_surrogate(x):
        return model.predict(actual_to_box(x).reshape(-1,2))[0][0,0]

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
        acc1 = np.zeros((nstep + 1, nvar), dtype=bool)  # Acceptance rates sur
        r1 = np.log(np.random.rand(nstep, nvar))  # Pre-computed random numbers for sur
        acc2 = np.zeros((nstep + 1, nvar), dtype=bool)  # Acceptance rates true
        r2 = np.log(np.random.rand(nstep, nvar))  # Pre-computed random numbers for true

        lpold_sur = -cost_surrogate(x[0, :])/(2.0*sig2meas)
        lpold = -cost(x[0, :], run_dir)/(2.0*sig2meas)
        for k in range(nwarm):
            x[k+1, :] = x[k, :]
            for i in range(nvar):
                xguess[:] = x[k+1, :]
                xguess[i] += dx[k, i]
                xguess[i] = np.abs(xguess[i])  # Mirror negative values
                lpnew_sur = -cost_surrogate(xguess)/(2.0*sig2meas)
                A_sur = lpnew_sur - lpold_sur
                if A_sur >= r1[k, i]:
                    acc1[k, i] = True
                else:  # Reject according to surrogate
                    continue

                lpnew = -cost(xguess, run_dir)/(2.0*sig2meas)
                A = lpnew - lpold - A_sur
                if A >= r2[k, i]:
                    x[k+1, :] = xguess
                    lpold = lpnew
                    lpold_sur = lpnew_sur
                    acc2[k, i] = True

        acceptance_rate = np.sum(acc2[:nwarm], 0)/nwarm
        target_rate = 0.35
        dx = dx*np.exp(acceptance_rate/target_rate-1.0)
        if ki < niwarm:
            x[0, :] = x[nwarm, :]

    plt.figure()
    plt.plot(x[:nwarm, 0], x[:nwarm, 1])
    plt.title(f'warmup, acceptance rate: {acceptance_rate}')
    plt.savefig(os.path.join(run_dir, '1.png'))

    print('Warmup acceptance rate: ', acceptance_rate)


    for k in range(nwarm, nstep):
        x[k+1, :] = x[k, :]
        for i in range(nvar):
            xguess[:] = x[k+1, :]
            xguess[i] += dx[k, i]
            lpnew_sur = -cost_surrogate(xguess)/(2.0*sig2meas)
            A_sur = lpnew_sur - lpold_sur
            if A_sur >= r1[k, i]:
                acc1[k, i] = True
            else:  # Reject according to surrogate
                continue

            lpnew = -cost(xguess, run_dir)/(2.0*sig2meas)
            A = lpnew - lpold - A_sur
            if A >= r2[k, i]:
                x[k+1, :] = xguess
                lpold = lpnew
                lpold_sur = lpnew_sur
                acc2[k, i] = True
    toc = time() - tic

    plt.figure()
    plt.plot(x[:, 0], x[:, 1])
    plt.plot(x[:nwarm, 0], x[:nwarm, 1])
    plt.title(f'MC, acceptance rates: \
        {np.sum(acc1[nwarm+1:], 0)/(nmc+1),np.sum(acc2[nwarm+1:], 0)/(nmc+1)}')
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
    np.save(f'xmc_da_{kproc}', x)

if __name__ == '__main__':
    main()
