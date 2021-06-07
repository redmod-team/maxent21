#%%
import numpy as np
import matplotlib.pyplot as plt
from algae_common import *

years = range(1997, 2001+1)
nyears = len(years)

T_water = []

# for year in years:
#     if year == 1997:
#         place = 'SCHNACKENBURG'
#     else:
#         place = 'GEESTHACHT'
#     file_water_temp = os.path.join(
#         template_dir, 'INPUT_DATA', place, f'WaterTemp_{year}.txt')
#     T_water.append(read_data(file_water_temp, f'{year}-03-12', f'{year}-10-29'))


for year in years:
    place = 'NEU_DARCHAU'
    file_water_temp = os.path.join(
        template_dir, 'INPUT_DATA', place, f'Discharge_{year}.txt')
    T_water.append(read_data(file_water_temp, f'{year}-03-12', f'{year}-10-29'))

#%%
Y = np.empty((nyears, len(T_water[0])))
for k, T in enumerate(T_water):
    T.plot()
    Y[k, :] = T.values.flat

# %%
from profit.sur.linear_reduction import KarhunenLoeve
kl = KarhunenLoeve(Y, tol=1e-12)
plot_kl(kl)

# %% Testing to express last year in terms of previous ones
kl = KarhunenLoeve(Y[:-1,:], tol=1e-12)
plot_kl(kl)

# %% Express new function in terms of existing features
znew = kl.project(Y[-1,:].reshape(1,-1))
ynew_approx = kl.lift(znew)

fig, ax = plt.subplots()
ax.plot(kl.ymean, 'k--')
ax.plot(ynew_approx[0,:])
ax.plot(Y[-1,:])
#%%
np.mean((Y[-1,:] - ynew_approx[0,:])**2)
np.mean((Y[-1,:] - kl.ymean)**2)
#%%
# %%
