import numpy as np

from sklearn.datasets import make_sparse_spd_matrix
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg, spsolve
from ilupp import solve
from time import time


N = 1000

A = csc_matrix(make_sparse_spd_matrix(N, alpha=0.99))
b = np.random.rand(N)

tic = time()
x = spsolve(A, b, use_umfpack=True)
print('Warmup UMFPACK: ', time() - tic, ' s, err = ', np.sqrt(np.mean((A @ x - b)**2)))

tic = time()
x = spsolve(A, b, use_umfpack=True)
print('UMFPACK: ', time() - tic, ' s, err = ', np.sqrt(np.mean((A @ x - b)**2)))

tic = time()
res = cg(A, b, tol=1e-14)
x = res[0]
print('CG: ', time() - tic, ' s, err = ', np.sqrt(np.mean((A @ x - b)**2)))

tic = time()
x = solve(A, b, max_iter=1000, rtol=1e-14, atol=1e-14)
print('ilupp: ', time() - tic, ' s, err = ', np.sqrt(np.mean((A @ x - b)**2)))
