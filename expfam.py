import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

sq2 = np.sqrt(2.0)
x = np.linspace(-3, 3, 101)
th = np.linspace(1, 3, 5)

def p(x, th):
    return np.exp(-np.abs(x/sq2)**(th))/(2.0*sq2*gamma(1.0 + 1.0/th))

plt.figure()
for thk in th:
    plt.plot(x, p(x, thk))

plt.legend(th)
