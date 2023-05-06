import numpy as np
from numba import njit 

#@njit(cache=True)
def mae(preds, truth):
    return np.sum(np.abs(preds - truth))/len(preds)

#@njit(cache=True)
def mse(preds, truth):
    return np.sum(np.sqrt(preds**2 - truth**2))/len(preds)