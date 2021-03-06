import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

sq2 = np.sqrt(2.0)
x = np.linspace(-3, 3, 101)
th = np.linspace(1, 3, 5)

def p(x, th):
    return np.exp(-np.abs(x/sq2)**(th))/(2.0*sq2*gamma(1.0 + 1.0/th))

plt.figure(figsize=(4.0,3.0))
for thk in th:
    plt.plot(x, p(x, thk))
plt.xlabel(r'$y$')
plt.ylabel(r'$p(y)$')
plt.legend(th)
plt.tight_layout()
#plt.savefig('paper/fig/dists.pdf')
#%%
from sympy import plot_implicit, symbols, Eq, And
x, y = symbols('x y')

from sympy import plot_implicit, symbols, Eq, And
p1 = plot_implicit(Eq(x**2 + y**2, 5), line_color='tab:blue', line_width=2)
# %%
