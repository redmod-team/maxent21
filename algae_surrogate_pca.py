#%%
import numpy as np
import matplotlib.pyplot as plt
import GPy
from GPy.models import GPRegression
from algae_common import *

file_meas = os.path.join(
    template_dir, 'INPUT_DATA', 'GEESTHACHT', 'Chla_Fluor_2001.txt')
data_meas = pd.read_csv(
    file_meas, delim_whitespace=True, header=0, skiprows=[1],
    na_values=['empty'], parse_dates=[0], dayfirst=True)
data_meas = data_meas.groupby('Date').mean().interpolate(
    limit_direction='both')
data_meas = data_meas[data_meas.index >= '2001-03-12']
data_meas = data_meas[data_meas.index <= '2001-10-29']

yref = fac_meas*fac_norm*data_meas['Chla_Fluor'].values
nt = len(yref)
plt.figure()
plt.plot(yref)


def residuals_y(y):
    return yref - y

def cost_y(y):
    return np.sum(residuals_y(y)**2, 1)/nt

X = np.load('Xtrain0.npy')
y = np.load('ytrain0.npy')
#%%
from profit.sur.linear_reduction import KarhunenLoeve
from sklearn.linear_model import LinearRegression
kl = KarhunenLoeve(y, tol=1e-13)

regr = LinearRegression()
j = np.arange(5, 20) + 1
regr.fit(np.log(j).reshape(-1, 1), np.log(kl.w[-6:-21:-1]).reshape(-1, 1))
print(regr.coef_)
print(regr.intercept_)

fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.loglog(1, kl.w[-1]/kl.w[-1], 'x')
for k in range(50):
    ax.loglog(k+1, kl.w[-k-1]/kl.w[-1], 'x')
ax.loglog([4, 50],
    np.exp(regr.predict(np.log([4, 50]).reshape(-1, 1))).flat/kl.w[-1], 'k')
ax.set_xlabel('Index')
ax.set_ylabel('Eigenvalues')
fig.tight_layout()



fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.plot(kl.ymean)
ax.plot(-kl.features()[:,::-1])
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$g(\tau)$')
ax.legend(['mean'] + [f'$\\varphi_{k+1}$' for k in range(3)], loc='upper right')
fig.tight_layout()

neig0 = len(kl.w)
kl.w = kl.w[neig0-neig:]
kl.Q = kl.Q[:, neig0-neig:]
ztrain = kl.project(y)

ykl = kl.lift(ztrain)
ktest = 9
plt.figure()
plt.plot(y[ktest, :])
plt.plot(ykl[ktest, :], '--')
#plt.plot(ymu[ktest, :])

#%%

models = []
for k, zk in enumerate(ztrain):
    kernel = GPy.kern.Matern52(2, ARD=True, lengthscale=10.0/nsamp0, variance=1)
    mf = GPy.mappings.Linear(2, 1)
    model = GPRegression(X, zk.reshape(-1, 1), kernel,
        noise_var=1e-4, mean_function=mf)
    model.optimize('bfgs')
    print(model.kern.lengthscale)
    models.append(model)
    model.save(f'model_{k}.hdf5')

# %%
def surrogate(x):
    mus = np.empty((neig, 1))
    for k, model in enumerate(models):
        mu, _ = model.predict(x.reshape(-1, 2), full_cov=False)
        mus[k, :] = mu.flat

    return kl.lift(mus)

def cost_surrogate(r):
    return cost_y(surrogate(actual_to_box(r)))

# %%
ktest = 9
plt.figure()
plt.plot(y[ktest, :])
plt.plot(ykl[ktest, :], '--')
plt.plot(surrogate(X[ktest,:]).T)

# %%
