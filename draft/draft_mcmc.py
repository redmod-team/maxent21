import numpy as np
import matplotlib.pyplot as plt


def pgauss(x):
    return np.exp(-x**2/2.0)


def step(x, dx, r, p):
    xguess = x + dx
    A = p(xguess)/p(x)
    if A >= 1:
        return xguess, True
    if A >= r:
        return xguess, True
    return x, False


nwarm = 500
nmc = 1000

nstep = nwarm + nmc
x = np.empty(nstep + 1)
acc = np.empty(nstep + 1, dtype=bool)
dx = np.random.rand(nstep) - 0.5
r = np.random.rand(nstep)

x[0] = 0.0
for k in range(nwarm):
    x[k+1], acc[k] = step(x[k], dx[k], r[k], pgauss)

plt.figure()
plt.plot(x[:nwarm])
plt.title('Warmup')

acceptance_rate = np.sum(acc[:nwarm])/nwarm
target_rate = 0.35
dx = dx*np.exp(acceptance_rate/target_rate-1.0)

for k in range(nwarm + 1, nstep):
    x[k+1], acc[k] = step(x[k], dx[k], r[k], pgauss)

plt.figure()
plt.plot(x)
plt.plot(x[:nwarm])
plt.title(f'MC, acceptance rate: {np.sum(acc[nwarm+1:])/(nmc+1)}')

plt.figure()
xref = np.random.randn(nmc)
plt.hist(x[nwarm+1:])
plt.hist(xref, alpha=0.5)
plt.title('Distribution')

c = np.correlate(x[nwarm+1:], x[nwarm+1:], mode='full')
cref = np.correlate(xref, xref, mode='full')
plt.figure()
plt.plot(c[len(c)//2:][:20])
plt.plot(cref[len(cref)//2:][:20])
plt.title('Autocorrelation')

print('Mean: ', x[nwarm+1:].mean())
print('Variance: ', x[nwarm+1:].var(ddof=1))  # Unbiased variance
