import numpy as np

def mae(preds, truth):
    return np.sum(np.abs(preds - truth))/len(preds)

def mse(preds, truth):
    return np.sum(np.sqrt(preds**2 - truth**2))/len(preds)