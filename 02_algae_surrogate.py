import numpy as np
import matplotlib.pyplot as plt
import GPy
from algae_common import *

plt.figure()
plt.plot(yref)

X = np.load('Xtrain0.npy')
y = np.load('ytrain0.npy')

model = GPy.models.GPRegression(X, cost_y(y).reshape(-1,1), k,
        noise_var=1e-4, mean_function=mf)
model.optimize('bfgs')

print(model)
print(model.kern.lengthscale)

model.save('sur0.hdf5')
