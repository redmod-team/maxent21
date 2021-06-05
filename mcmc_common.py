import numpy as np

def mcmc(x0, dx, niwarm, nwarm, nmc, cost, cost_surrogate=None):
    if cost_surrogate:  # Delayed acceptance
        return mcmc_da(x0, dx, niwarm, nwarm, nmc, cost, cost_surrogate)

    nvar = len(x0)
    nstep = nwarm + nmc
    xguess = np.empty(nvar)
    x = np.empty((nstep + 1, nvar))
    x[0, :] = x0

    for ki in range(niwarm):
        acc = np.zeros((nstep + 1, nvar), dtype=bool)  # Acceptance rates
        r = np.log(np.random.rand(nstep, nvar))  # Pre-computed random numbers

        lpold = -cost(x[0, :]) # log-likelihood
        for k in range(nwarm):
            x[k+1, :] = x[k, :]
            for i in range(nvar):
                xguess[:] = x[k+1, :]
                xguess[i] += dx[k,i]
                lpnew = -cost(xguess)
                A = lpnew - lpold
                if A >= r[k,i]:
                    x[k+1, :] = xguess
                    lpold = lpnew
                    acc[k,i] = True

        acceptance_rate = np.sum(acc[:nwarm], 0)/nwarm
        print('Warmup acceptance rate: ', acceptance_rate)

        target_rate = 0.35
        dx = dx*np.exp(acceptance_rate/target_rate-1.0)
        if ki < niwarm:
            x[0, :] = x[nwarm, :]


    lpold = -cost(x[nwarm, :])  # log-likelihood
    for k in range(nwarm, nstep):
        x[k+1, :] = x[k, :]
        for i in range(nvar):
            xguess[:] = x[k+1, :]
            xguess[i] += dx[k,i]
            lpnew = -cost(xguess)
            A = lpnew - lpold
            if A >= r[k, i]:
                x[k+1, :] = xguess
                lpold = lpnew
                acc[k,i] = True

    return x, acc


def mcmc_da(x0, dx, niwarm, nwarm, nmc, cost, cost_surrogate):
    nvar = len(x0)
    nstep = nwarm + nmc
    xguess = np.empty(nvar)
    x = np.empty((nstep + 1, nvar))
    x[0, :] = x0

    # Warmup

    for ki in range(niwarm):
        acc1 = np.zeros((nstep + 1, nvar), dtype=bool)  # Acceptance rates sur
        r1 = np.log(np.random.rand(nstep, nvar))  # Pre-computed random numbers for sur
        acc2 = np.zeros((nstep + 1, nvar), dtype=bool)  # Acceptance rates true
        r2 = np.log(np.random.rand(nstep, nvar))  # Pre-computed random numbers for true

        lpold_sur = -cost_surrogate(x[0, :])
        lpold = -cost(x[0, :])
        for k in range(nwarm):
            x[k+1, :] = x[k, :]
            for i in range(nvar):
                xguess[:] = x[k+1, :]
                xguess[i] += dx[k, i]
                lpnew_sur = -cost_surrogate(xguess)
                A_sur = lpnew_sur - lpold_sur
                if A_sur >= r1[k, i]:
                    acc1[k, i] = True
                else:  # Reject according to surrogate
                    continue

                lpnew = -cost(xguess)
                A = lpnew - lpold - A_sur
                if A >= r2[k, i]:
                    x[k+1, :] = xguess
                    lpold = lpnew
                    lpold_sur = lpnew_sur
                    acc2[k, i] = True

        acceptance_rate = np.sum(acc2[:nwarm], 0)/nwarm
        print('Warmup acceptance rate: ', acceptance_rate)

        target_rate = 0.35
        dx = dx*np.exp(acceptance_rate/target_rate-1.0)

        if ki < niwarm:
            x[0, :] = x[nwarm, :]


    for k in range(nwarm, nstep):
        x[k+1, :] = x[k, :]
        for i in range(nvar):
            xguess[:] = x[k+1, :]
            xguess[i] += dx[k, i]
            lpnew_sur = -cost_surrogate(xguess)
            A_sur = lpnew_sur - lpold_sur
            if A_sur >= r1[k, i]:
                acc1[k, i] = True
            else:  # Reject according to surrogate
                continue

            lpnew = -cost(xguess)
            A = lpnew - lpold - A_sur
            if A >= r2[k, i]:
                x[k+1, :] = xguess
                lpold = lpnew
                lpold_sur = lpnew_sur
                acc2[k, i] = True

    return x, acc1, acc2
