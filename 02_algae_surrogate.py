import numpy as np
import matplotlib.pyplot as plt
import GPy
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

k = GPy.kern.Matern52(nvar, ARD=True, lengthscale=1.0, variance=1)
mf = GPy.mappings.Linear(nvar, 1)
#mf = GPy.mappings.Constant(nvar, 1)

X = np.load('Xtrain0.npy')
y = np.load('ytrain0.npy')

model = GPy.models.GPRegression(X, cost_y(y).reshape(-1,1), k,
        noise_var=1e-4, mean_function=mf)
model.optimize('bfgs')

print(model)
print(model.kern.lengthscale)

model.save('sur0.hdf5')
