import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import profit
import GPy
from subprocess import run

template_dir = os.path.expanduser('~/src/DiatomGrowth/profit/template')
base_dir = '/dev/shm/calbert/vecma21'
exe = os.path.expanduser('~/src/DiatomGrowth/main.x')
home_dir = os.path.expanduser('~/run/algae/vecma21')

with open(os.path.join(template_dir, 'input.txt'), 'r') as f:
    template = f.read()

keys = ['K_light', 'mu_0', 'f_Si', 'lambda_S', 'a', 'sigma_0']
mean = np.array([41.9, 1.19, 0.168, 0.0118, 1.25, 0.15])
maxval = np.array([500.0, 3.5, 0.4, 0.05, 2.0, 2.0])

nvar = 2
ntout = 232
nsamp0 = 128

fac_norm = 1.0/200.0  # To get values of O(1)
fac_meas = 5.2  # Conversion from Chla_Fluor to Chlorphyl concentration

sig_ch = 5.0  # Basic uncertainty
sig2meas = (sig_ch*fac_norm)**2  # Measurement variance

x0test = np.array([447.0, 1.9])

neig = 10

def init_dir(run_dir):
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(os.path.join(run_dir, 'RESULTS'))
    shutil.copytree(os.path.join(template_dir, 'INPUT_DATA'),
        os.path.join(run_dir, 'INPUT_DATA'), symlinks=False)


def read_data(filename, date1='2001-03-12', date2='2001-10-29'):
    data_meas = pd.read_csv(
        filename, delim_whitespace=True, header=0, skiprows=[1],
        na_values=['empty'], parse_dates=[0], dayfirst=True)
    data_meas = data_meas.groupby('Date').mean().interpolate(
        limit_direction='both')
    data_meas = data_meas[data_meas.index >= date1]
    data_meas = data_meas[data_meas.index <= date2]
    return data_meas


def blackbox(x, run_dir):
    params = {'year': 2001}
    for k, key in enumerate(keys):
        if (k < nvar):
            params[key] = x[k]
        else:
            params[key] = mean[k]

    input = profit.pre.replace_template(template, params)

    with open(os.path.join(run_dir, 'input.txt'), 'w') as f:
        f.write(input)

    run(exe, shell=False, text=False, capture_output=True, cwd=run_dir)

    y = fac_norm*np.loadtxt(
        os.path.join(run_dir, 'RESULTS', 'results.txt'), skiprows=1)

    return y

def box_to_actual(x):
    return x*maxval[:nvar]

def actual_to_box(r):
    return r/maxval[:nvar]

def plot_kl(kl):
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.semilogy(1, kl.w[-1]/kl.w[-1], 'x')
    for k in range(len(kl.w)):
        ax.semilogy(k+1, kl.w[-k-1]/kl.w[-1], 'x')
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalues')
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(kl.ymean)
    ax.plot(-kl.features()[:,::-1])
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$g(\tau)$')
    ax.legend(['mean'] + [f'$\\varphi_{k+1}$' for k in range(3)])
    fig.tight_layout()


data_meas = read_data(
    os.path.join(template_dir, 'INPUT_DATA', 'GEESTHACHT',
        'Chla_Fluor_2001.txt'))

yref = fac_meas*fac_norm*data_meas['Chla_Fluor'].values

k = GPy.kern.Matern52(nvar, ARD=False, lengthscale=0.2, variance=1)
#mf = GPy.mappings.Linear(nvar, 1)
mf = GPy.mappings.Constant(nvar, 1)

def residuals(x, run_dir):
    return yref - blackbox(x, run_dir)

def cost(x, run_dir):
    return np.sum(residuals(x, run_dir)**2)*1.0/ntout

def residuals_y(y):
    return yref - y

def cost_y(y):
    return np.sum(residuals_y(y)**2, 1)*1.0/ntout


thstar = np.array([500.0, 3.5, 0.4, 0.05, 2.0, 2.0])
Pstar = 0.9
b = thstar/np.tan(np.pi/2*Pstar)

def prior(x):
    return 1.0/np.pi*b[:nvar]/(b[:nvar]**2 + x**2)
