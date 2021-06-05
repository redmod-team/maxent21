import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 101)
th = np.linspace(1, 3, 5)

def p(x, th):
    return np.exp(-np.abs(x)**(th))

plt.figure()
for thk in th:
    plt.plot(x, p(x, thk))

plt.legend(th)
