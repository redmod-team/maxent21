# %%
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from profit.util.halton import halton
from multiprocessing import Process
from algae_common import *


nproc = 16

def main():
    workers = []
    jobs_per_worker = nsamp0//nproc
    X = np.empty((nproc, jobs_per_worker, nvar))
    y = np.zeros((nproc, jobs_per_worker, ntout))


    # Sample in circle segment
    r = halton(4096, nvar)
    r = r[np.sum(r**2, 1) < 1.0, :]
    r = r[:nsamp0, :]

    rundirs = []
    for kproc in range(nproc):
        run_dir = os.path.join(base_dir, f'{kproc}')
        shutil.rmtree(run_dir, ignore_errors=True)
        os.makedirs(os.path.join(run_dir, 'RESULTS'))
        shutil.copytree(os.path.join(template_dir, 'INPUT_DATA'),
            os.path.join(run_dir, 'INPUT_DATA'), symlinks=False)

        rundirs.append(os.path.join(base_dir, f'{kproc}'))
        X[kproc, :, :] = r[kproc*jobs_per_worker:(kproc+1)*jobs_per_worker]
        workers.append(Process(target=start_run,
            args=(run_dir, X[kproc,:,:])))
        workers[-1].start()

    for kproc, worker in enumerate(workers):
        worker.join()
        y[kproc, :] = np.load(os.path.join(base_dir, f'{kproc}', 'y.npy'))

    np.save('Xtrain0.npy', X.reshape(nsamp0, nvar))
    np.save('ytrain0.npy', y.reshape(nsamp0, -1))

def start_run(run_dir, X):
    y = np.array([blackbox(box_to_actual(xk), run_dir) for xk in X])
    np.save(os.path.join(run_dir, 'y.npy'), y)

if __name__ == '__main__':
    main()
