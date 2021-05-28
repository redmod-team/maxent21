# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import profit
from scipy.optimize import leastsq
from scipy.special import erfinv
from subprocess import run
from profit.util.halton import halton

template_dir = os.path.expanduser('~/src/DiatomGrowth/profit/template')
run_dir = os.path.expanduser('~/run/algae/vecma21')
exe = os.path.expanduser('~/src/DiatomGrowth/main.x')

with open(os.path.join(template_dir, 'input.txt'), 'r') as f:
    template = f.read()

print(template)

mean = {
    'K_light':  41.9,
    'mu_0':     1.19,
    'f_Si':     0.168,
    'lambda_S': 0.0118,
    'a':        1.25,
    'sigma_0':  0.15
}

stdev = {
    'K_light':  10,
    'mu_0':     0.2,
    'f_Si':     0.02,
    'lambda_S': 0.002,
    'a':        0.2,
    'sigma_0':  0.02
}

r = halton(2, 6)
rn = {k: mean[k] + stdev[k]*np.sqrt(2.0)*erfinv(2.0*r[:, i] - 1.0)
      for i, k in enumerate(mean)}

def blackbox(x):
    params = {'year': 2001}
    for k, key in enumerate(mean):
        params[key] = x[k]

    input = profit.pre.replace_template(template, params)

    with open(os.path.join(run_dir, 'input.txt'), 'w') as f:
        f.write(input)

    run(exe, shell=True, text=True, cwd=run_dir)

    return np.loadtxt(
        os.path.join(run_dir, 'RESULTS', 'results.txt'), skiprows=1)

# %%
ropt = list(mean.values())
rstart = [v[0] for v in rn.values()]

yopt = blackbox(ropt)

def residuals(x):
    return yopt - blackbox(x)

xopt, cov_x, infodict, mesg, ier = leastsq(
    residuals, x0=rstart, full_output=True)

# %%
plt.figure()
plt.plot(yopt)
plt.plot(blackbox(rstart))
plt.plot(blackbox(xopt))
plt.legend(['reference', 'start', 'optimized'])

# %%
